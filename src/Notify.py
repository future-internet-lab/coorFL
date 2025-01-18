import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_mail(config, body: str):
    if not config["sender-email"]:
        return

    sender_email = config["sender-email"]
    password = config["password"]
    receiver_email = config["receiver-email"]

    subject = "Thông báo về quá trình training"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print("Email đã được gửi thành công!")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
