import json
from typing import Any

import duckdb
import numpy as np
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse, Response

from loop.api.models import (
    ApplyPlanRequest,
    BriefResponse,
    ClassifyRequest,
    ClassifyResponse,
    CustomerItem,
    CustomerListResponse,
    CustomerSignalsResponse,
    HealthResponse,
    InboxActionRequest,
    InboxResponse,
    LeadsTodayResponse,
    LedgerResponse,
    PlanResponse,
    RepliesResponse,
    ReplyListItem,
    RunWeekResponse,
    SignalItem,
)
from loop.brief.builder import build_customer_brief
from loop.config import ARTIFACTS_DIR, DB_PATH, REPORTS_DIR
from loop.ops.inbox import fetch_inbox_data, record_inbox_action
from loop.ops.leads import fetch_today_leads, generate_leads_csv
from loop.ops.plan import apply_operator_plan, fetch_operator_plan
from loop.ops.simulation import execute_next_week_simulation, fetch_ledger_data
from loop.reply_intel.baseline import BaselineClassifier, train_baseline
from loop.reply_intel.calibration import apply_temperature
from loop.reply_intel.drafts import generate_draft_response
from loop.reply_intel.llm_classifier import classify_with_llm
from loop.reply_intel.validators import run_all_validators
from loop.schema import REPLY_CLASSES, ReplyClass, init_db

router = APIRouter()

def get_db_con() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        from loop.simulate.campaigns import run_simulation
        run_simulation()
    init_db(DB_PATH)
    return duckdb.connect(str(DB_PATH), read_only=False)

@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(status="ok")

@router.get("/customers", response_model=CustomerListResponse)
def get_customers() -> CustomerListResponse:
    con = get_db_con()
    rows = con.execute("SELECT customer_id, name, industry, primary_language, n_sends FROM customers ORDER BY customer_id;").fetchall()
    con.close()
    customers = [
        CustomerItem(
            customer_id=str(r[0]),
            name=str(r[1]),
            industry=str(r[2]),
            primary_language=str(r[3]),
            n_sends=int(r[4]),
        )
        for r in rows
    ]
    return CustomerListResponse(customers=customers)

@router.get("/customers/{customer_id}/brief", response_model=BriefResponse)
def get_customer_brief(customer_id: str) -> BriefResponse:
    con = get_db_con()
    res = con.execute("SELECT brief_json FROM briefs WHERE customer_id = ?;", [customer_id]).fetchone()
    if res and res[0]:
        data = json.loads(res[0])
        con.close()
        return BriefResponse(**data)

    rw_con = init_db(DB_PATH)
    try:
        brief_data = build_customer_brief(customer_id, rw_con)
        brief_json = json.dumps(brief_data)
        rw_con.execute("INSERT OR REPLACE INTO briefs (customer_id, brief_json) VALUES (?, ?);", [customer_id, brief_json])
        rw_con.close()
        con.close()
        return BriefResponse(**brief_data)
    except Exception as exc:
        rw_con.close()
        con.close()
        raise HTTPException(status_code=404, detail=str(exc))

@router.get("/customers/{customer_id}/signals", response_model=CustomerSignalsResponse)
def get_customer_signals(customer_id: str) -> CustomerSignalsResponse:
    con = get_db_con()
    cust_row = con.execute("SELECT name, industry FROM customers WHERE customer_id = ?;", [customer_id]).fetchone()
    if not cust_row:
        con.close()
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    name, ind = str(cust_row[0]), str(cust_row[1])

    rows = con.execute(
        "SELECT key, median, lo90, hi90, extra_json FROM posterior_summaries WHERE customer_id = ? AND quantity = 'signal_lift_age0';",
        [customer_id],
    ).fetchall()
    con.close()

    signal_items: list[SignalItem] = []
    for r in rows:
        sig_name = str(r[0])
        med = float(r[1])
        lo = float(r[2])
        hi = float(r[3])
        extra = json.loads(r[4]) if r[4] else {}

        decay_curve = extra.get("decay_curve", [])
        hl_med = extra.get("half_life_median", 0.0)
        hl_lo = extra.get("half_life_lo90", 0.0)
        hl_hi = extra.get("half_life_hi90", 0.0)
        naive = extra.get("naive_lift", med)

        signal_items.append(
            SignalItem(
                signal=sig_name,
                median_lift=round(med, 2),
                lo90=round(lo, 2),
                hi90=round(hi, 2),
                half_life_median=round(hl_med, 1),
                half_life_lo90=round(hl_lo, 1),
                half_life_hi90=round(hl_hi, 1),
                naive_lift=round(naive, 2),
                decay_curve=decay_curve,
            )
        )

    signal_items.sort(key=lambda s: s.median_lift, reverse=True)

    if signal_items:
        top_sig = signal_items[0]
        finding_title = f"{top_sig.signal.replace('_', ' ').title()} produces +{top_sig.median_lift:.2f}pp positive lift at age 0 with {top_sig.half_life_median:.0f}-day half-life"
    else:
        finding_title = "No signal lift data available"

    return CustomerSignalsResponse(
        customer_id=customer_id,
        customer_name=name,
        industry=ind,
        finding_title=finding_title,
        signals=signal_items,
    )

@router.get("/customers/{customer_id}/replies", response_model=RepliesResponse)
def get_customer_replies(
    customer_id: str,
    cls: str | None = Query(None),
    language: str | None = Query(None),
    needs_review: bool | None = Query(None),
    q: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> RepliesResponse:
    con = get_db_con()

    query = """
    SELECT 
        r.reply_id,
        r.send_id,
        r.customer_id,
        r.received_at,
        r.text,
        r.true_class,
        r.true_language,
        COALESCE(c.predicted_class, r.true_class) as predicted_class,
        COALESCE(c.confidence, 0.95) as confidence,
        COALESCE(c.routed_to_review, false) as routed_to_review,
        c.referral_name,
        c.referral_email,
        c.revisit_date,
        c.draft_reply,
        c.validators_json,
        c.class_probs_json
    FROM replies r
    LEFT JOIN classifications c ON r.reply_id = c.reply_id
    WHERE r.customer_id = ?
    """
    params: list[Any] = [customer_id]

    if cls:
        query += " AND (r.true_class = ? OR c.predicted_class = ?)"
        params.extend([cls, cls])
    if language:
        query += " AND r.true_language = ?"
        params.append(language)
    if needs_review is not None:
        query += " AND c.routed_to_review = ?"
        params.append(needs_review)
    if q:
        query += " AND r.text ILIKE ?"
        params.append(f"%{q}%")

    query += " ORDER BY r.received_at DESC;"

    df = con.execute(query, params).fetchdf()
    con.close()

    total = len(df)
    auto_handled = int((~df["routed_to_review"]).sum())
    review_count = int(df["routed_to_review"].sum())

    df_page = df.iloc[offset : offset + limit]

    replies: list[ReplyListItem] = []
    for _, row in df_page.iterrows():
        validators = json.loads(row["validators_json"]) if row["validators_json"] else {}
        class_probs = json.loads(row["class_probs_json"]) if row["class_probs_json"] else {}

        replies.append(
            ReplyListItem(
                reply_id=str(row["reply_id"]),
                send_id=str(row["send_id"]),
                customer_id=str(row["customer_id"]),
                received_at=str(row["received_at"]),
                text=str(row["text"]),
                true_class=str(row["true_class"]),
                true_language=str(row["true_language"]),
                predicted_class=str(row["predicted_class"]),
                confidence=float(row["confidence"]),
                routed_to_review=bool(row["routed_to_review"]),
                referral_name=row["referral_name"],
                referral_email=row["referral_email"],
                revisit_date=row["revisit_date"],
                draft_reply=row["draft_reply"],
                validators=validators,
                class_probabilities=class_probs,
            )
        )

    return RepliesResponse(
        total=total,
        limit=limit,
        offset=offset,
        auto_handled_count=auto_handled,
        review_count=review_count,
        replies=replies,
    )

@router.post("/replies/classify", response_model=ClassifyResponse)
def classify_reply_endpoint(req: ClassifyRequest) -> ClassifyResponse:
    llm_res = classify_with_llm(req.text, req.received_at)
    if llm_res is not None:
        pred_cls: ReplyClass = llm_res["predicted_class"]
        conf = float(llm_res["confidence"])
        lang = str(llm_res.get("language", "en"))
        ref_name = llm_res.get("referral_name")
        ref_email = llm_res.get("referral_email")
        revisit_date = llm_res.get("revisit_date")
        probs = llm_res["probabilities"]
        model_name = llm_res["model"]
    else:
        model_path = ARTIFACTS_DIR / "baseline_model.joblib"
        if model_path.exists():
            clf = BaselineClassifier.load(model_path)
        else:
            clf = train_baseline()
            clf.save(model_path)

        raw_probs = clf.predict_proba([req.text])
        calib_file = ARTIFACTS_DIR / "classifier_calibration.json"
        temp = 1.0
        tau = 0.70
        if calib_file.exists():
            try:
                with open(calib_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    temp = float(cdata.get("temperature", 1.0))
                    tau = float(cdata.get("tau", 0.70))
            except Exception:
                pass

        calib_probs = apply_temperature(raw_probs, temp)[0]
        best_i = int(np.argmax(calib_probs))
        pred_cls = REPLY_CLASSES[best_i]
        conf = float(calib_probs[best_i])
        probs = {REPLY_CLASSES[j]: float(calib_probs[j]) for j in range(len(REPLY_CLASSES))}

        from langdetect import detect
        try:
            lang = detect(req.text)
        except Exception:
            lang = "en"

        ref_name = None
        ref_email = None
        revisit_date = None
        model_name = "baseline_calibrated"

    calib_file = ARTIFACTS_DIR / "classifier_calibration.json"
    tau = 0.70
    if calib_file.exists():
        try:
            with open(calib_file, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                tau = float(cdata.get("tau", 0.70))
        except Exception:
            pass

    routed = bool(conf < tau)
    draft = generate_draft_response(req.text, pred_cls, lang, ref_name, ref_email)
    validators_res = run_all_validators(draft, pred_cls, lang, req.text, ref_name, ref_email)

    return ClassifyResponse(
        predicted_class=pred_cls,
        confidence=conf,
        probabilities=probs,
        language=lang,
        referral_name=ref_name,
        referral_email=ref_email,
        revisit_date=revisit_date,
        draft_reply=draft,
        validators=validators_res,
        routed_to_review=routed,
        model=model_name,
    )

@router.get("/evaluation")
def get_evaluation_results() -> Any:
    report_file = REPORTS_DIR / "eval_results.json"
    if not report_file.exists():
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Evaluation results not found. Please run 'make eval' or 'make eval-fast' to generate evaluation artifacts."},
        )
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)

@router.get("/customers/{customer_id}/inbox", response_model=InboxResponse)
def get_inbox_endpoint(customer_id: str) -> InboxResponse:
    con = get_db_con()
    data = fetch_inbox_data(customer_id, con)
    con.close()
    return InboxResponse(**data)

@router.post("/replies/{reply_id}/action")
def reply_action_endpoint(reply_id: str, req: InboxActionRequest, customer_id: str = Query(...)) -> Any:
    res = record_inbox_action(
        reply_id=reply_id,
        customer_id=customer_id,
        action=req.action,
        final_text=req.final_text,
        corrected_class=req.corrected_class,
    )
    return res

@router.get("/customers/{customer_id}/leads/today", response_model=LeadsTodayResponse)
def get_today_leads_endpoint(customer_id: str, limit: int = Query(40, ge=1, le=100)) -> LeadsTodayResponse:
    data = fetch_today_leads(customer_id, limit=limit)
    return LeadsTodayResponse(**data)

@router.get("/customers/{customer_id}/leads/export.csv")
def export_leads_csv_endpoint(customer_id: str) -> Response:
    csv_str = generate_leads_csv(customer_id)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=leads_{customer_id}.csv"},
    )

@router.get("/customers/{customer_id}/plan", response_model=PlanResponse)
def get_plan_endpoint(customer_id: str) -> PlanResponse:
    data = fetch_operator_plan(customer_id)
    return PlanResponse(**data)

@router.post("/customers/{customer_id}/plan/apply")
def apply_plan_endpoint(customer_id: str, req: ApplyPlanRequest) -> Any:
    return apply_operator_plan(customer_id, req.shares)

@router.get("/customers/{customer_id}/ledger", response_model=LedgerResponse)
def get_ledger_endpoint(customer_id: str) -> LedgerResponse:
    data = fetch_ledger_data(customer_id)
    return LedgerResponse(**data)

@router.post("/customers/{customer_id}/weeks/run", response_model=RunWeekResponse)
def run_next_week_endpoint(customer_id: str) -> RunWeekResponse:
    res = execute_next_week_simulation(customer_id)
    return RunWeekResponse(**res)

