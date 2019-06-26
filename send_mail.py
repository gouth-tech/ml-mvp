import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from Mail.fetch_mail import username
from classification import predict_from_text, text


def sendmail():
    prediction = predict_from_text(str(text))
    message = Mail(
        from_email=username,
        to_emails='goutham.m@hashroot.com',
        subject='Classification',
        html_content='<strong>' + prediction + '</strong>')
    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        sg.send(message)
        print("A message sent to goutham.m@hashroot.com")
    except Exception as e:
        print(str(e))


sendmail()
