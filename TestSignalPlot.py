from matplotlib import pyplot as plt
import numpy as np
import math
import scipy as sp

# For the purposes of this we will use a singal power of unity


mod_mode = {'BPSK':1, 'QPSK':2, '8PSK':3, '16QAM':4, '64QAM':6}

transmit_carrier_freq = 10
class Modulator:
    def __init__(self, input = "Hello World!", carrier_freq = transmit_carrier_freq, mod_mode_select = 'BPSK'):
        self.input = input
        self.carrier_freq = carrier_freq
        self.mod_mode_select = mod_mode_select
        self.digitalSig = np.array([])
        self.graphres = (1/self.carrier_freq)/1000
        self.check = False

    def GenerateSignals(self):
        symbolrate = mod_mode[self.mod_mode_select]
        period = 1/carrier_freq
        graphtime, bitstr = self.UserChar2BitStream(self.input, self.carrier_freq, self.mod_mode_select)
        digitaltransmission = self.DigitalSignal(bitstr, self.carrier_freq)

        match self.mod_mode_select:
            case 'BPSK':
                modulated = self.BPSKModulation(bitstr, self.carrier_freq)
            case 'QPSK':
                modulated = self.QPSKModulation(bitstr, self.carrier_freq)
            case '8PSK':
                modulated = self.PSK8Modulation(bitstr, self.carrier_freq)
            case '16QAM':
                modulated = self.QAM16Modulation(bitstr, self.carrier_freq)
            case '64QAM':
                modulated = self.QAM64Modulation(bitstr, self.carrier_freq)

        modulated = np.pad(modulated, (0, len(graphtime) - len(modulated)), 'constant', constant_values=(0,0))
        digitaltransmission = np.pad(digitaltransmission, (0, len(graphtime) - len(digitaltransmission)), 'constant', constant_values=(0,0))

        return graphtime, digitaltransmission, modulated
    
    def ModulatedPlot(self):
        period = 1/carrier_freq
        graphtime, digitaltransmission, modulated = self.GenerateSignals()
        
        fig, (ax1,ax2) = plt.subplots(2, 1)
        ax1.plot(graphtime, digitaltransmission)
        ax1.set_title("Digital Signal")
        ax1.set_ylabel("Amplitude")

        ax1.vlines(x= graphtime[::int(period/self.graphres)], ymin = -0.5, ymax = 1.5, colors = 'r', linestyles = 'dashed',alpha=0.75)
        ax2.plot(graphtime, modulated)
        ax2.set_title("Modulated Signal")
        ax2.set_ylabel("Amplitude")
        ax2.vlines(x= graphtime[::int(period/self.graphres)], ymin = -1.5, ymax = 1.5, colors = 'r', linestyles = 'dashed',alpha=0.75)
        ax2.set_xlabel("Time (s)")
        plt.show()


    def UserChar2BitStream(self, input = "Hello World!",carrier_freq = transmit_carrier_freq, mod_mode_select = 'BPSK'):
        symbolrate = mod_mode[mod_mode_select] #Select the symbol rate based on the modulation mode
        
        strbytes = input.encode('utf-8') #Convert the input string to binary
        strbit = ''.join(f'{byte:08b}' for byte in strbytes) #Convert the binary string to a bit string

        period = (1/carrier_freq)/symbolrate #Calculate the time period of the carrier wave, translates to the time per symbol
        graph_time = np.arange(0, math.ceil(len(strbit)/10)*10*period + self.graphres, self.graphres) #Create the time axis for the graph
        
        return graph_time, strbit

    def DigitalSignal(self, bitstream, carrier_freq = transmit_carrier_freq):
        symbolrate = mod_mode[self.mod_mode_select] #Select the symbol rate based on the modulation mode
        period = 1/carrier_freq
        wavegenarr = np.arange(0, period/symbolrate, self.graphres)
        digitaltransmission = np.array([])
        for i in range(len(bitstream)):
            if bitstream[i] == '0':
                bit_sig = np.zeros(len(wavegenarr))
            else:
                bit_sig = np.ones(len(wavegenarr))
            digitaltransmission = np.append(digitaltransmission, bit_sig)
        return digitaltransmission
    
    def BPSKModulation(self, bitstream, carrier_freq = transmit_carrier_freq):
        assert len(bitstream) % 2 == 0, "BPSK requires an even number of bits"
        period = 1/carrier_freq #Calculate the time period of the carrier wave
        wavegenarr = np.arange(0, period, self.graphres) #Create the time axis for the carrier wave
        modulatedtransmission = np.array([])

        phasemap = {
            '01': 3 * np.pi / 4,  # 3π/4
            '11': np.pi / 4,      # π/4
            '00': 5 * np.pi / 4,  # 5π/4
            '10': 7 * np.pi / 4   # 7π/4
        } #Map the bitstream to phase shifts

        for i in range(len(bitstream),2):
            bit_pair = bitstream[i:i+2]
            mod_bit_sig = np.cos(2 * np.pi * carrier_freq * wavegenarr + phasemap[bit_pair]) #Modulate the bitstream
            modulatedtransmission = np.append(modulatedtransmission, mod_bit_sig)
        return modulatedtransmission
    
    def QPSKModulation(self, bitstream, carrier_freq = transmit_carrier_freq):
        assert len(bitstream) % 2 == 0, "QPSK requires an even number of bits"
        period = 1/carrier_freq
        wavegenarr = np.arange(0, period, self.graphres)
        modulatedtransmission = np.array([])
        Primary = bitstream[0::2]
        Quadrature = bitstream[1::2]
        for i in range(0, len(Primary)):
            mod_bit_sig = (np.sqrt(1/2))*(int(Primary[i])*2-1) * np.cos(2 * np.pi * carrier_freq * wavegenarr) + (np.sqrt(1/2))*(int(Quadrature[i])*2-1) * np.sin(2 * np.pi * carrier_freq * wavegenarr) #Modulate the bitstream in quadrature form
            modulatedtransmission = np.append(modulatedtransmission, mod_bit_sig)
        return modulatedtransmission

    def PSK8Modulation(self, bitstream, carrier_freq = transmit_carrier_freq):
        period = 1/carrier_freq
        wavegenarr = np.arange(0, period, self.graphres)
        modulatedtransmission = np.array([])
        bitstream = bitstream + '0'*(len(bitstream) % 3) #Pad the bitstream with zeros to make it divisible by 3
        Primary = bitstream[0::3]
        Quadrature = bitstream[1::3]
        Tertiary = bitstream[2::3]
        for i in range(0, len(Primary)):
            n = int(Primary[i] + Quadrature[i] + Tertiary[i], 2)
            mod_bit_sig = (np.sqrt(1/2))*np.cos(2 * np.pi * carrier_freq * wavegenarr + (2*n-1) * np.pi / 8) + (np.sqrt(1/2))*np.sin(2 * np.pi * carrier_freq * wavegenarr + (2*n-1) * np.pi / 8)
            modulatedtransmission = np.append(modulatedtransmission, mod_bit_sig)
        return modulatedtransmission

if __name__ == "__main__":
    inputmessage = input("Enter the message to be transmitted: ")
    carrier_freq = float(input("Enter the carrier frequency: "))
    try:
        mod_mode_select = input("Enter the modulation mode (BPSK, QPSK, 8PSK, 16QAM, 64QAM): ").upper()
        mod_test = mod_mode[mod_mode_select]
    except:
        print("Invalid modulation mode. Defaulting to BPSK.")
        mod_mode_select = 'BPSK'

    Modulator = Modulator(inputmessage, carrier_freq, mod_mode_select)
    Modulator.ModulatedPlot()



