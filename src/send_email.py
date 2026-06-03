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
from email.mime.text import MIMEText

from dotenv import load_dotenv


# 项目根目录：investment_os
BASE_DIR = Path(__file__).resolve().parents[1]

# .env 文件路径
ENV_FILE = BASE_DIR / ".env"


def send_email(subject: str, body: str) -> None:
    """
    发送一封纯文本邮件。

    参数：
        subject: 邮件标题
        body: 邮件正文
    """

    # 读取 .env 文件
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

    # 创建邮件正文
    msg = MIMEText(body, "plain", "utf-8")

    # 邮件标题
    msg["Subject"] = subject

    # 发件人
    msg["From"] = gmail_user

    # 收件人
    msg["To"] = email_to

    # 使用 Gmail SMTP SSL 端口
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        # 登录 Gmail
        server.login(gmail_user, gmail_app_password)

        # 发送邮件
        server.sendmail(
            gmail_user,
            [email_to],
            msg.as_string(),
        )


if __name__ == "__main__":
    send_email(
        subject="Investment OS Test Email",
        body="This is a test email from Investment OS.",
    )

    print("Test email sent successfully.")