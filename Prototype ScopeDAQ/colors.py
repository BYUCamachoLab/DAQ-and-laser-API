#A quick little hack to change the channel colors on the scope.
#Sends colors in RGB format.

RED     = (255, 255, 000, 000)
GREEN   = (255, 000, 255, 000)
BLUE    = (255, 000, 000, 255)

channel_colors = {
    # Channel: (Alpha, Red, Green, Blue) Range 0 to 255.
    1: (255, 000, 000, 000),
    2: GREEN,
    3: BLUE,
    4: RED
}

#some_cool_colors = {
#    0xffff49fb,
#    0xff31ff99,
#    0xffc8c8c8,
#    0xff3060ff
#}

MIN_VALUE = 0
MAX_VALUE = 255

def waveformSuffix(channelNum):
    if 1 <= channelNum <= 4:
        return 3*(channelNum - 1) + 2
    else:
        raise Exception("Channel number {} out of range (1-4).".format(channelNum))

def connectScope():
    import visa
    rm = visa.ResourceManager()
    scope = rm.open_resource("TCPIP::10.32.112.140::hislip0")
    return scope

def getHex(number):
    if MIN_VALUE <= number <= MAX_VALUE:
        return hex(number)[-2:]
    else:
        raise Exception("Number is out of range (0-255).")


#ARGB: AARRGGBB (Alpha, Red, Green, Blue). Hexadecimal representation.
def getARGB(alpha, red, green, blue):
    hexString = "0x{}{}{}{}".format(
        getHex(alpha),
        getHex(red),
        getHex(green),
        getHex(blue)
    )
    return int(hexString, 16)

def sendColor(scope, channel, color_data):
    scope.write("DISPLAY:COLOR:SIGNAL{}:COLOR {}".format(
        waveformSuffix(channel),
        getARGB(color_data)
    ))
