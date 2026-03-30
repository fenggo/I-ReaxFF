import smtplib
from email.mime.text import MIMEText
# from dingtalkchatbot.chatbot import DingtalkChatbot
from os import popen,getcwd
import re


# webhook='https://oapi.dingtalk.com/robot/send?access_token=9a567ecf401bbd43a7b63bafe36c30f5c88b660a7831c326ad4f6291c9460cce'
# dlam   = DingtalkChatbot(webhook)

class DLAM(object):
  def __init__(self):
      ''' fake DingtalkChatbot ''' 

  def send_markdown(self,title='Learning Information: ',text=' '):
      print('------------------------------------------------------------------------')
      print(' * ',title)
      # print(' ')
      print(' * ',text)
      print('------------------------------------------------------------------------')

dlam = DLAM()


def valid_ip(ip):
    if ("255" in ip) or ( ip == "127.0.0.1") or ( ip == "0.0.0.0" ) or ( ip == "127.0.1.1"):
        return False
    else:
        return True

# def get_ip(valid_ip):
#     ipss = ''.join(popen('ifconfig').readlines())
#     match = "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
#     ips = re.findall(match, ipss, flags=re.M)
#     ip = filter(valid_ip, ips)
#     IP = ''
#     for i,I in enumerate(ip):
#         if i==0:
#            IP = I
#         else:
#            IP += '/'+I
#     return IP

# ip = get_ip(valid_ip)
# hn=''.join(popen('hostname').readlines())
# hostname = hn[:-1]


def send_msg(msg):
    cwd=getcwd()
    # msg = '#### MY Master:\n'+'----------------\n '+ msg
    # msg += '\n\n----------------\n'
    # msg += '\n From: '+hostname+'@'+ip+'\n' #\nmy master
    msg += '\n *  (directory: '+cwd+')'
    # try:
    dlam.send_markdown(title='Learning Information: ',text=msg)
    # except:
    # send_mail(msg)
    # print('-  warning massage send failed.')

