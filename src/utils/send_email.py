"""
send_email.py

作用：
读取 .env 里的 Gmail 配置，然后发送一封测试邮件。

这个文件之后会被 weekly_summary.py 调用，
用于把周报自动发送到你的邮箱。
"""

from pathlib import Path
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv


# 项目根目录：investment_os
BASE_DIR = Path(__file__).resolve().parents[2]

# .env 文件路径
ENV_FILE = BASE_DIR / ".env"


def _configured_email_settings() -> tuple[str, str, list[str]]:
    """Load and validate the existing Gmail environment configuration."""

    load_dotenv(ENV_FILE)

    gmail_user = os.getenv("GMAIL_USER")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")
    email_to = os.getenv("EMAIL_TO")

    if not gmail_user:
        raise ValueError("Missing GMAIL_USER in .env")

    if not gmail_app_password:
        raise ValueError("Missing GMAIL_APP_PASSWORD in .env")

    if not email_to:
        raise ValueError("Missing EMAIL_TO in .env")

    recipients = [
        recipient.strip()
        for recipient in email_to.replace(";", ",").split(",")
        if recipient.strip()
    ]
    if not recipients:
        raise ValueError("EMAIL_TO in .env contains no recipients")
    return gmail_user, gmail_app_password, recipients


def send_email(
    subject: str,
    body: str,
    *,
    html_body: str | None = None,
) -> int:
    """
    使用现有 Gmail 配置发送纯文本或 multipart/alternative 邮件。

    参数：
        subject: 邮件标题
        body: 邮件正文
        html_body: 可选 HTML 正文；提供时保留纯文本 fallback

    返回：
        实际配置的收件人数。
    """

    gmail_user, gmail_app_password, recipients = (
        _configured_email_settings()
    )
    if html_body is None:
        msg = MIMEText(body, "plain", "utf-8")
    else:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

    # 邮件标题
    msg["Subject"] = subject

    # 发件人
    msg["From"] = gmail_user

    # 收件人
    msg["To"] = ", ".join(recipients)

    # 使用 Gmail SMTP SSL 端口
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        # 登录 Gmail
        server.login(gmail_user, gmail_app_password)

        # 发送邮件
        server.sendmail(
            gmail_user,
            recipients,
            msg.as_string(),
        )
    return len(recipients)


if __name__ == "__main__":
    send_email(
        subject="Investment OS Test Email",
        body="This is a test email from Investment OS.",
    )

    print("Test email sent successfully.")
