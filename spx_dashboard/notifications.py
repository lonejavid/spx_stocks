"""
Notifications for SPX Day Trading Dashboard.
Telegram bot alert when signal score >= 6.
"""
import requests


def send_telegram_alert(
    bot_token: str,
    chat_id: str,
    signal_type: str,
    score: int,
    spx_price: float,
    time_est: str,
    triggers: list,
) -> bool:
    """
    Send a Telegram message when STRONG signal (score 6+).
    triggers = list of condition strings that were met.
    Returns True if sent successfully.
    """
    if not bot_token or not chat_id:
        return False
    triggers_str = ", ".join(triggers[:5]) if triggers else "N/A"
    msg = (
        f"{signal_type} - SPX @ ${spx_price:,.2f}\n"
        f"Score: {score}/8 conditions met\n"
        f"Time: {time_est}\n"
        f"Key triggers: {triggers_str}"
    )
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": msg, "disable_web_page_preview": True},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False
