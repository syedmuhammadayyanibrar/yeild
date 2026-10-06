# Reply Annotation Guide and Taxonomy

## 1. Overview
Loop classifies incoming sales replies into 10 mutually exclusive classes. Positive replies consist strictly of `interested` and `info_request`.

## 2. Taxonomy Classes
1. `out_of_office`: Automated out-of-office, vacation, sabbatical, or leave notifications.
2. `unsubscribe`: Direct opt-out requests, remove requests, cease and desist, or complaints about unsolicited contact.
3. `referral`: Reply directing the sender to a colleague or alternative contact person, often with contact details.
4. `wrong_person`: Stating the recipient is not the right person for this function, without providing a specific referral.
5. `objection_timing`: Deferring evaluation to a future quarter, year, or planning cycle (e.g., "Check back in Q3").
6. `objection_budget`: Citing lack of budget, hiring freeze, or spending freeze.
7. `objection_competitor`: Already using a competing vendor or locked into an existing contract.
8. `not_relevant`: Generic mismatch with the company's tech stack, needs, or domain without specific objection details.
9. `info_request`: Requesting additional documentation, one-pagers, case studies, or security questionnaires.
10. `interested`: Clear interest in seeing a demo, discussing capabilities, or scheduling a meeting.

## 3. Precedence Rules for Ambiguity
When a reply contains multiple signals or conflicting statements, label assignment resolves strictly according to the following precedence hierarchy:

`out_of_office` > `unsubscribe` > `referral` > `wrong_person` > `objection_*` (`objection_timing` = `objection_budget` = `objection_competitor`) > `info_request` > `interested`

### Examples
- "I am out of the office until Monday. Please remove me from your list." -> `out_of_office`
- "Please take me off your mailing list, but you could talk to Jane in procurement." -> `unsubscribe`
- "I don't handle this. Reach out to sarah@example.com." -> `referral`
- "Not my department." -> `wrong_person`
- "We have no budget this quarter, but send over a one-pager." -> `objection_budget`
- "Sounds interesting, can you send over some information first?" -> `info_request`
- "Let's connect for 15 minutes next Tuesday." -> `interested`
