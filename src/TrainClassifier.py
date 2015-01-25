import ConfigParser
from network import bufhelp
import time

Config = ConfigParser.ConfigParser()
Config.read("settings.ini")

def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

connectionOptions = ConfigSectionMap("Connection")

hostname = connectionOptions("hostname")
port = connectionOptions("port")
#connect to buffer
bufhelp.connect(hostname=hostname,port=port)
print("connected")

time.sleep(1)
print("train classifier START")
bufhelp.sendevent('startPhase.cmd', 'train')
while True:
    print("wait for classifier")
    e = bufhelp.waitforevent(("sigproc.training","end"), 1000, False)
    if e is not None:
        break
print("training END")
