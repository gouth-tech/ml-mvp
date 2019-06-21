import imaplib, getpass
import email


FROM_EMAIL = "tuser9378@gmail.com"
FROM_PWD = getpass.getpass()
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT = 993


def read_email_from_gmail():
    try:

        mail = imaplib.IMAP4_SSL(SMTP_SERVER)
        mail.login(FROM_EMAIL,FROM_PWD)
        mail.select('inbox')

        type, data = mail.search(None, 'All')
        mail_ids = data[0]

        id_list = mail_ids.split()
        first_email_id = int(id_list[-2])
        latest_email_id = int(id_list[-1])

        for i in range(latest_email_id,first_email_id, -1):
            typ, data = mail.fetch(i, '(RFC822)')
            for response_part in data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_string(response_part[1])
                    file = open('data.txt', "w+")
                    file.flush()
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":  # ignore attachments/html
                            body = part.get_payload(decode=True)
                            file.write(body)




    except Exception, e:
        print str(e)


read_email_from_gmail()
