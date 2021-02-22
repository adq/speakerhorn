#!/bin/python 

import numpy
import socket
import time
from rgbmatrix5x5 import RGBMatrix5x5


AUDIO_PACKET_SIZE = 8 * 1024
UDP_PORT = 5555
MID_FREQ_BAND_START_HZ = 250
HIGH_FREQ_BAND_START_HZ = 2000

# calculate array indexes that our three frequency bands start at
# bufsize / 2 / 2 'cos there are two bytes per audio sample and there are also two audio channels
midstartidx = 0
highstartidx = 0
fft_freqs = numpy.fft.rfftfreq(int(AUDIO_PACKET_SIZE / 2 / 2), 1 / 44100)
for i, f in enumerate(fft_freqs):
    if f < MID_FREQ_BAND_START_HZ:
        midstartidx = i
    elif f < HIGH_FREQ_BAND_START_HZ:
        highstartidx = i

# init the rgb matrix
rgbmatrix5x5 = RGBMatrix5x5()
rgbmatrix5x5.set_clear_on_exit()
rgbmatrix5x5.set_brightness(0.8)

# init the socket we receive data on
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', UDP_PORT))

# 1 second timeout so we can clear the display on stop
sock.settimeout(1)

# set the socket internal buffering to a single packet so we always get the latest sample and not a
# huge buffer of out of date data
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, AUDIO_PACKET_SIZE * 5)

# setup the PCM datatype for numpy -- S16L
pcm_dt = numpy.dtype(numpy.int16).newbyteorder('L')

# main loop
display_cleared = False
while True:
    # try and get some data, and swallow timeouts
    try:
        buf = sock.recv(AUDIO_PACKET_SIZE)
    except socket.timeout:
        buf = None

    # no data => just clear the display and keep going
    if not buf:
        if not display_cleared:
            rgbmatrix5x5.clear()
            rgbmatrix5x5.show()

            display_cleared = True
        continue

    # load in the raw data, interpreting it as little endian signed 16 bit
    pcm = numpy.frombuffer(buf, dtype=pcm_dt)

    # reshape the input array into two columns (since its interleaved L+R channels)
    pcm = numpy.reshape(pcm, (int(len(pcm) / 2), 2))

    # average the L+R channels together into a single value
    pcm = pcm.mean(axis=1)

    # perform an FFT to shift to frequency domain
    freq = numpy.fft.rfft(pcm)

    # calculate power of each individual frequency
    freq_power = numpy.square(numpy.abs(freq))

    # now calculate power of each frequency band, as well as the total frequency power for this data chunk
    low_band_power = numpy.sum(freq_power[: midstartidx])
    mid_band_power = numpy.sum(freq_power[midstartidx: highstartidx])
    high_band_power = numpy.sum(freq_power[highstartidx:])
    total_band_power = low_band_power + mid_band_power + high_band_power

    if total_band_power:
        # map the frequency bands to R(low) G(mid) and B(high) and update the display
        r = low_band_power / total_band_power
        g = mid_band_power / total_band_power
        b = high_band_power / total_band_power
        rgbmatrix5x5.set_all(int(r * 255), int(g * 255), int(b * 255))
        rgbmatrix5x5.show()
        display_cleared = False

    else:
        # ... unless the sound was all zero => just wipe the display rather than crashing
        if not display_cleared:
            rgbmatrix5x5.clear()
            rgbmatrix5x5.show()
            display_cleared = True

    # sleep a bit
    time.sleep(0.05)
