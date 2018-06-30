from numpy import ceil,hamming,log2,zeros,floor,shape,zeros,log,kron,ones,mean,hanning,conj,concatenate,sqrt,exp,sign
from scipy.special import expi
from numpy import divide,multiply,ones,array,reshape,sum,add
from numpy import matrix
import soundfile as sf
from numpy.fft import fft, ifft
def srec_fe_ag(wave, dif_mul, step, step0, per80, per20):
    alpha_n = 0.80
    alpha_s = 0.55
    Nmic = 5
    front_params_16000 = dict()
    front_params_16000['mel_dim'] = 12
    front_params_16000['samplerate'] = 16000
    front_params_16000['window_factor'] = 2.0
    front_params_16000['pre_mel'] = 0.899999976
    front_params_16000['low_cut'] = 125
    front_params_16000['high_cut'] = 5500
    front_params_16000['do_skip_even_frames'] = True
    front_params_16000['do_smooth_c0'] = True
    front_params_16000['do_dd_mel'] = True
    front_params_16000['peakpickup'] = 0.300000012
    front_params_16000['peakpickdown'] = 0.699999988
    front_params_16000['melA_scale'] = [ 14, 45, 60, 70, 95, 115, 115, 135, 135, 155, 160, 180 ]
    front_params_16000['melB_scale'] = [ 42, 110, 105, 110, 140, 140, 150, 120, 150, 130, 140, 130 ]
    front_params_16000['dmelA_scale'] = [50, 150, 290, 320, 400, 500, 500, 600, 600, 700, 720, 750]
    front_params_16000['dmelB_scale'] = [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127]
    front_params_16000['ddmelA_scale'] = [4, 12, 22, 27, 32, 35, 35, 45, 45, 55, 57, 62]
    front_params_16000['ddmelB_scale'] = [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127]
    front_params_16000['voice_margin'] = 14
    front_params_16000['fast_voice_margin'] = 18
    front_params_16000['tracker_margin ']= 7
    front_params_16000['voice_duration'] = 6
    front_params_16000['quiet_duration'] = 20

    params = front_params_16000

    framerate = 100

    params['n_features'] = 12

    frame_period = int(params['samplerate'] / framerate)
    window_length = int(floor(params['window_factor'] * frame_period))

    window = hamming(window_length)

    bandwidth = params['samplerate'] / 2
    params['np'] = int(2 ** ceil(log2(window_length)))
    #params['cut_off_below'] = (params['low_cut'] * params.np) / (2.0 * bandwidth) + 1
    #params['cut_off_above'] = (params['high_cut'] * params.np) / (2.0 * bandwidth) + 1




    n_frames = int(floor(shape(wave)[1]/ frame_period) - ceil(params['window_factor']) - 1)



    phin = zeros((257, 1))
    phip = zeros((257, 1))

    summ_all = zeros((257, int(n_frames)))

    mag_sqr_next = zeros((257, 1))
    mag_sqr_dif_next = zeros((257, 1))
    phase_next = zeros((257, 1))


    ldif_mul = log(dif_mul)
    eps = 1e-10


    f0 = 9
    f1 = 109
    wave = array (wave)
    for i in range (n_frames):


        frame = wave[:, i* frame_period : i*frame_period+window_length]
        windowed = frame * kron(ones((Nmic,1)), window)
        fd_frame = fft(windowed, params['np'])

        half_fd_frame = fd_frame[:,0:int(params['np'] / 2 + 1)]
        summ = mean(half_fd_frame, 1)


        phase_next = [s/(abs(s)+eps) for s in summ]
        mag_sqr = mag_sqr_next
        mag_sqr_dif = mag_sqr_dif_next
        sum0 = array([abs(s)**2 for s in summ])
        print(sum0.shape)
        mag_sqr_next = array( [4 * s for s in sum0])

        dif0 = zeros((257, 1))

        ss = 0
        dd = 0

        for ii  in  range(Nmic-1):
            for j  in range (ii+1,Nmic):

                dd_ij = abs(half_fd_frame[:][ii] - half_fd_frame[:][j])**2
                ss_ij = abs(half_fd_frame[:][ii] + half_fd_frame[:][j])** 2
                dif0 = [dd_ij[i]+dif0[i] for i in range(len(dd_ij))]

                dd = dd + sum(dd_ij[f0:f1])
                ss = ss + sum(ss_ij[f0:f1])

        dif0 = array(dif0)
        crit = log([s+ eps for s in sum0]) - log([d+eps for d in dif0])

        crit0 = mean(crit[f0:f1])
        per80 = per80 + step0 * (sign(crit0 - per80) + 2 * 0.8 - 1)
        per20 = per20 + step0 * (sign(crit0 - per20) + 2 * 0.2 - 1)




        dif_mul = exp(ldif_mul);

        dif = dif0 * dif_mul;

        if Nmic == 0:
            dif = 0


        mag_sqr_dif_next = 4 * dif

        print(mag_sqr.shape, mag_sqr_next.shape)
        pp = [0.5 * (mag_sqr[j] + mag_sqr_next[j]) for j in range(len(mag_sqr))]
        pn = [0.5 * (mag_sqr_dif[j] + mag_sqr_dif_next[j]) for j in range(len(mag_sqr_dif))]
        print(pp.shape)
        phip = alpha_s * phip + (1 - alpha_s) * pp

        phin = alpha_n * phin + (1 - alpha_n) * pn

        phin1 = phin

        ln_sig = 0.35 * (log(abs(phin1) + eps)) - 6.0266

        phase = phase_next
        phase_next = summ / (abs(summ) + eps)

        reg = eps
        print (phip[1])
        dev =  divide([max(eps,phip[i]-phin1[i]+reg) for i in range(len(phip))],(abs(phip) + eps))
        wiener = array([min(1 - eps,dev[i]) for i in range (len(dev))])


        nsr = [w ** -1 - 1 for w in wiener]
        prob = [1/(1+2*n) for n in nsr]
        log_voice_est = log(mag_sqr + eps) + 2 * log(wiener) + expi(nsr **-1)
        mag_sqr = exp((log_voice_est - ln_sig)* prob + ln_sig)
        mag_sqr_nr = mag_sqr

        mag_sqr = mag_sqr_next
        mag_sqr_dif = mag_sqr_dif_next
        sum0 = abs(summ)**2
        mag_sqr_next = 4 * sum0
        summ_all[:][i] = sqrt(mag_sqr_nr)* phase

    summ_all = summ_all[:][1:len(summ_all)]
    wave_est = zeros((n_frames * 160, 1))
    for i in range (0,n_frames):
        af_half = summ_all[:][i]
        af = concatenate(af_half,conj(reversed(af_half[1:len(af_half)])))
        a = ifft(af)
        wave_est[(i - 1) * frame_period + 1: (i - 1) * frame_period + window_length][1] = wave_est[(i - 1) * frame_period + window_length][1] + a[0: window_length] / window* hanning(window_length)

    return wave_est


if __name__ == "__main__":

    ch1 ,fs = array(sf.read('simu/F01_22GC010A_BUS.CH1.wav'))
    ch3, fs = array(sf.read('simu/F01_22GC010A_BUS.CH3.wav'))
    ch4, fs = array(sf.read('simu/F01_22GC010A_BUS.CH4.wav'))
    ch5, fs = array(sf.read('simu/F01_22GC010A_BUS.CH5.wav'))
    ch6, fs = array(sf.read('simu/F01_22GC010A_BUS.CH6.wav'))
    step = 1e-2
    step0 = 1e-4
    per80 = -3
    per20 = -4
    Nmic = 5
    dif_mul0 = multiply(ones((257, 1)) ,(1 / (Nmic ** 2 * (Nmic - 1))))
    dif_mul = dif_mul0

    wave = [ch1,ch3,ch4,ch5,ch6]
    wave_est = srec_fe_ag(wave,dif_mul,step,step0,per80,per20)