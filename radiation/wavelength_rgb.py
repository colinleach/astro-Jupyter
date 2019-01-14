def wavelength_RGB(wlen):
    """ wlen: wavelength in nm
        needs single value, np.array fails
        
        returns: (R,G,B) triplet of integers (0-255)
    
        Credits: Dan Bruton http://www.physics.sfasu.edu/astro/color.html"""
    
    # first pass at an RGB mix
    if 380 <= wlen and wlen < 440:
        red = (440 - wlen) / (440 - 380)
        green = 0.0
        blue = 1.0
    elif 440 <= wlen and wlen < 490:
        red = 0.0
        green = (wlen - 440)/(490 - 440)
        blue = 1.0
    elif 490 <= wlen and wlen < 510:
        red = 0.0
        green = 1.0
        blue = (510 - wlen) / (510 - 490)
    elif 510 <= wlen and wlen < 580:
        red = (wlen - 510)/(580 - 510)
        green = 1.0
        blue = 0.0
    elif 580 <= wlen and wlen < 645:
        red = 1.0
        green = (645 - wlen)/(645 - 580)
        blue = 0.0
    elif 645 <= wlen and wlen < 780:
        red = 1.0
        green = 0.0
        blue = 0.0
    else:
        red = 0.0
        green = 0.0
        blue = 0.0
       
    # reduce brightness towards the extremes where our eyes are less sensitive
    if 380 <= wlen and wlen < 420:
        factor = 0.3 + 0.7 * (wlen - 380)/(420 - 380)
    elif 420 <= wlen and wlen < 700:
        factor = 1.0
    elif 700 <= wlen and wlen < 780:
        factor = 0.3 + 0.7 * (780 - wlen)/(780 - 700)
    else:
        factor = 0.0
        
    gamma = 0.8
    intensity_max = 255
    R = int(intensity_max * (red * factor)**gamma)
    G = int(intensity_max * (green * factor)**gamma)
    B = int(intensity_max * (blue * factor)**gamma)
    
    return (R,G,B)
