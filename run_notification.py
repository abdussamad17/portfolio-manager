import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import datetime
import sys

today = datetime.datetime.now().date()

load_dotenv()
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')

def send_notification_email(notification_message):
    # Define the email addresses
    sender_email = "abdussamadusman99@gmail.com"
    receiver_email = "abdussamadusman99@gmail.com"

    # Create the email subject and body
    subject = str(today)
    body = notification_message

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

# Get the notification message from the command line argument
notification_message = sys.argv[1] if len(sys.argv) > 1 else "No specific message"
# Call the function to send the email
send_notification_email(notification_message)
