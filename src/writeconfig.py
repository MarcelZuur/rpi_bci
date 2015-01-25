import ConfigParser


Config = ConfigParser.ConfigParser(allow_no_value = True)
# lets create that config file for next time...
cfgfile = open("settings.ini", 'w')

# add the settings to the structure of the file, and lets write it out...
Config.add_section('Connection')
Config.set("Connection", "; Use your hostname or IP adress")
Config.set('Connection', 'hostname', "127.0.0.1")
Config.set('Connection', 'port', 1972)
Config.add_section('LED')
Config.set("LED", "; Set the GPIO pin numbering and led flickering frequencies")
Config.set("LED", "gpio", [13, 15, 11])
Config.set("LED", "frequencies", [11, 13, 9])
Config.add_section('Calibration')
Config.set("Calibration", "; Modify the calibration sequence")
Config.set("Calibration", "sequences", 6)
Config.set("Calibration", "interEpochDelay", 0.5)
Config.set("Calibration", "stimulusDuration", 2.5)
Config.set("Calibration", "stimulusEventDelay", 0.5)
Config.set("Calibration", "stimulusFullDuration", 2)
Config.set("Calibration", "interSequenceDelay", 5)
Config.add_section('SignalProcessing')
Config.set("SignalProcessing", "fsample", 256)
Config.set('SignalProcessing', '; SVM for Linear Support Vector Classification or LR for Logistic Regression')
Config.set("SignalProcessing", "classifier", "SVM")
Config.set("SignalProcessing", "channels", range(9, 22))
Config.write(cfgfile)
cfgfile.close()