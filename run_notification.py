import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import datetime

today = datetime.datetime.now().date()
log_file_path = f"../../logs/{today}.log"

load_dotenv()
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')

def send_notification_email():
    # Define the email addresses
    sender_email = "abdussamadusman99@gmail.com"
    receiver_email = "abdussamadusman99@gmail.com"

    # Read the log file content
    try:
        with open(log_file_path, 'r') as file:
            log_content = file.read()
    except FileNotFoundError:
        log_content = f"No log file found for {today}"

    # Create the email subject and body
    subject = f"Process Completion Notification on {today}"
    body = f"The process has been completed successfully on {today}\n\nLog Content:\n{log_content}"

    # Create the email object
    email = MIMEText(body)
    email['From'] = sender_email
    email['To'] = receiver_email
    email['Subject'] = subject

    # Define the AWS SES SMTP server settings
    smtp_server = "email-smtp.us-east-1.amazonaws.com"
    smtp_port = 587
    smtp_username = SMTP_USERNAME # SMTP user name
    smtp_password = SMTP_PASSWORD # SMTP password

    # Connect to the AWS SES SMTP server and send the email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, receiver_email, email.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        server.quit()

# Call the function to send the email
send_notification_email()
