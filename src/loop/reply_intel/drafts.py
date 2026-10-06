from loop.schema import ReplyClass

DRAFT_TEMPLATES: dict[str, dict[ReplyClass, str]] = {
    "en": {
        "interested": "Great to connect! You can book directly on my calendar: https://cal.com/demo or let me know if Tuesday 10am or Thursday 2pm works better.",
        "info_request": "Thanks for your interest. Attached is our product overview and architecture guide. Please let me know what questions you have.",
        "referral": "Thank you for pointing me in the right direction. I will reach out to {referral_name} at {referral_email} directly.",
        "wrong_person": "Understood, thank you for clarifying. Could you point me to who might be the right person or correct contact for this on your team?",
        "objection_timing": "Understood. I will follow up then as suggested. Best of luck with your current priorities.",
        "objection_budget": "Understood. We will keep in touch for future cycles when circumstances align.",
        "objection_competitor": "Makes sense, thank you for letting me know. If anything changes down the road, feel free to reach back out.",
        "not_relevant": "Thank you for the clarification. I appreciate your time and have noted this accordingly.",
        "unsubscribe": "You have been removed from our distribution lists and will not receive further outreach.",
        "out_of_office": "",
    },
    "fr": {
        "interested": "Merci pour votre retour. Vous pouvez choisir un créneau sur mon lien de calendrier : https://cal.com/demo ou me confirmer si mardi 10h ou jeudi 14h vous convient.",
        "info_request": "Merci pour votre intérêt. Vous trouverez ci-joint notre documentation de synthèse. Je reste à votre écoute pour toute question.",
        "referral": "Merci beaucoup pour cette indication. Je vais contacter {referral_name} sur {referral_email} de ce pas.",
        "wrong_person": "C'est bien noté. Pourriez-vous m'indiquer qui serait la bonne personne à contacter pour ce sujet ?",
        "objection_timing": "C'est bien noté, je reviendrai vers vous à la période indiquée. Bon courage pour vos priorités actuelles.",
        "objection_budget": "Bien compris. Nous reprendrons contact lors de votre prochain exercice budgétaire.",
        "objection_competitor": "C'est bien noté. Si vos besoins venaient à évoluer, n'hésitez pas à nous recontacter.",
        "not_relevant": "Merci pour votre retour précis. J'ai bien pris note de ces éléments.",
        "unsubscribe": "Votre adresse a été supprimée de nos listes de diffusion et vous ne recevrez plus de messages.",
        "out_of_office": "",
    },
    "de": {
        "interested": "Vielen Dank für Ihre Rückmeldung. Gerne können Sie direkt einen Termin buchen: https://cal.com/demo oder mir mitteilen, ob Dienstag 10 Uhr oder Donnerstag 14 Uhr passt.",
        "info_request": "Vielen Dank für Ihr Interesse. Anbei sende ich Ihnen unser Datenblatt und die Übersicht zu unserer Plattform.",
        "referral": "Vielen Dank für den Hinweis. Ich werde mich direkt an {referral_name} unter {referral_email} wenden.",
        "wrong_person": "Verstanden, vielen Dank. Könnten Sie mir kurz mitteilen, wer in Ihrem Team der richtige Ansprechpartner für diesen Bereich ist?",
        "objection_timing": "Vielen Dank für die Information. Ich melde mich gerne zum genannten Zeitpunkt wieder bei Ihnen.",
        "objection_budget": "Verstanden. Wir halten Sie für künftige Budgetzyklen gerne auf dem Laufenden.",
        "objection_competitor": "Vielen Dank für die Rückmeldung. Sollte sich künftig etwas ändern, freuen wir uns über eine Nachricht.",
        "not_relevant": "Vielen Dank für Ihre Klarstellung. Ich habe dies entsprechend vermerkt.",
        "unsubscribe": "Ihre E-Mail-Adresse wurde erfolgreich aus unserem Verteiler ausgetragen.",
        "out_of_office": "",
    },
    "es": {
        "interested": "Muchas gracias por su respuesta. Puede agendar directamente en mi enlace de calendario: https://cal.com/demo o avisarme si le viene bien martes 10:00 o jueves 14:00.",
        "info_request": "Gracias por su interés. Le adjunto la documentación técnica y el resumen de la solución para su revisión.",
        "referral": "Muchas gracias por la referencia. Contactaré directamente a {referral_name} en {referral_email}.",
        "wrong_person": "Entendido, muchas gracias. ¿Podría indicarme quién es la persona o contacto adecuado para este ámbito en su equipo?",
        "objection_timing": "Entendido, retomaremos el contacto en las fechas indicadas. Éxitos con sus proyectos actuales.",
        "objection_budget": "Comprendido perfectamente. Estaremos en contacto para siguientes periodos presupuestarios.",
        "objection_competitor": "Entendido, muchas gracias por hacérmelo saber. Quedamos a su disposición si en el futuro lo necesitan.",
        "not_relevant": "Muchas gracias por la aclaración. He dejado nota al respecto en nuestro sistema.",
        "unsubscribe": "Su dirección de correo ha sido dada de baja de nuestras listas de distribución.",
        "out_of_office": "",
    },
}

def generate_draft_response(
    reply_text: str,
    reply_class: ReplyClass,
    language: str,
    referral_name: str | None = None,
    referral_email: str | None = None,
) -> str | None:
    if reply_class == "out_of_office":
        return None

    lang_key = language if language in DRAFT_TEMPLATES else "en"
    raw_template = DRAFT_TEMPLATES[lang_key].get(reply_class, "")

    ref_name = referral_name if referral_name else "the team"
    ref_email = referral_email if referral_email else "the provided address"

    draft = raw_template.replace("{referral_name}", ref_name).replace("{referral_email}", ref_email)
    return draft
