import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from IPython.display import display, Math, Latex
from matplotlib import animation, rc

def calculate_wave_height(z, dx):
    top_peaks, _ = find_peaks(z, height = z.mean()*1.1)
    bottom_peaks, _ = find_peaks(-1.*z, height = z.mean()*-0.8, distance = int(1.2/dx))

    ztop = [z[i] for i in top_peaks]
    zbottom = [z[i] for i in bottom_peaks]

    ztop = np.array(ztop)
    ztop = np.delete(ztop, ztop.argmin())
    
    zbottom = np.array(zbottom)
    zbottom = np.delete(zbottom, zbottom.argmax())
    

    return ztop.mean() - zbottom.mean()
                
            
    
def calculate_wave_length(z,dx):
    #peaks, _ = find_peaks(-1.*z, height = z.mean()*-1., distance = int(1.2/dx))
    
    top_peaks, _ = find_peaks(z, height = z.mean()*1.1)
    
    lengths = []
    last_peak = None
    for peak in top_peaks:
        if last_peak == None:
            last_peak = peak
        else:
            
            tmpLength = dx * (peak - last_peak)
            
            if tmpLength > 0.4:
                lengths.append(tmpLength)
                last_peak = peak
                
    lengths = np.array(lengths)
    lengths = np.delete(lengths, lengths.argmin())
    lengths = np.delete(lengths, lengths.argmax())
    return lengths
    
def calculate_wave_movement(last_peak_index, current_z, dx):
    
    peaks, _ = find_peaks(current_z, height = current_z.mean()*0.8, distance = int(1.2/dx))
    
    updated_peak = None
    for peak in peaks:
        if peak > last_peak_index:
            updated_peak = peak
            #print(peak, last_peak_index)
            break

    return updated_peak

def calculate_wave_speed(verts, dx, dt, step_index = 4, base_index = 0):
    last_peak_index = None
    velocities = []
    timesteps = []
    
    
    for t in range(0, (verts.shape[0]), step_index):
        current_z = verts[t,:,1]
        peaks, _ = find_peaks(current_z, height = current_z.mean()*0.8, distance = int(1.2/dx))
        
        if last_peak_index is not None:
            delta = peaks[base_index] - last_peak_index[base_index]
            
            if delta == 0:
                raise ValueError('Delta was zero')
            if delta > 0:
                velocities.append(delta*dx/(dt*step_index))
            else:
                velocities.append(velocities[len(velocities)-1])
            
            timesteps.append( dt * t )
            #print(delta, delta*dx/(dt*step_index), dt * t)
        last_peak_index = peaks
    return velocities, timesteps




def plot_results(verts, dx, dt, extractionTime):
    
    extractionTimeMins = extractionTime/60.
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,15))

    heights = [calculate_wave_height(verts[t,:,1],dx) for t in range((verts.shape[0]))]
    times = np.array([t*extractionTimeMins for t in range((verts.shape[0]))])
    
    spl_height = UnivariateSpline(times, heights, k=4)
    ts = np.linspace(0, times.max(), len(verts))

    #heights_filtered = savgol_filter(heights,81,2)
    heights_filtered = savgol_filter(heights, len(heights)/2, 2)

    #ax1.plot(ts, spl_height(ts))
    ax1.plot(times, heights_filtered)
    #ax1.plot(times,heights,'o')
    ax1.set_xlabel('Time (mins)')
    ax1.set_ylabel('$\Delta$ (m)')
    x = verts[0,:,0]
    dx = x[1] - x [0]
    lengths = [calculate_wave_length(verts[t,:,1],dx).mean() for t in range((verts.shape[0]))]
    
    
    spl_lengths = UnivariateSpline(times, lengths, k=4)
    ts = np.linspace(0, times.max(), len(verts))
    
    ax2.plot(ts, spl_lengths(ts))
    ax2.set_xlabel('Time (mins)')
    ax2.set_ylabel('$\Lambda$ (m)')
    ax2.set_ylim([0, 3.0])

    for t in range(0, (verts.shape[0]),25):
        z = verts[t,:,1]
        h = verts[t,:,4]
        x = verts[t,:,0]
        timestep = t * 5.
        #ax3.plot(x,z, label='{0} mins'.format(timestep))
        #ax3.plot(x,z+h, 'b')

    #ax3.legend()
    ax3.set_xlabel('t (mins)')
    ax3.set_ylabel('$\delta$ (-)')
    
    
    # Steepness
    steepness = np.zeros(len(lengths))
    for i in range(len(lengths)):
        steepness[i] = heights[i]/lengths[i]
    
    spl_steepness = UnivariateSpline(times, steepness, k=4)
    ts = np.linspace(0, times.max(), len(verts))
    
    ax3.plot(times, spl_steepness(ts))    
    

    # this is in m/min
    v, timesteps = calculate_wave_speed(verts, dx, extractionTimeMins)
    v = [vs*60. for vs in v]
    timesteps=np.array(timesteps)
    
    spl_vs = UnivariateSpline(timesteps, v, k=2)
    ts = np.linspace(0, timesteps.max(), len(verts))
    
    v_filtered = savgol_filter(v,25,2)
    
    #ax4.plot(ts,spl_vs(ts), '.')
    #ax4.plot(timesteps,v, '.')
    ax4.plot(timesteps, v_filtered)
    #ax4.set_ylim([0, 0.5])

    ax4.set_xlabel('Time (mins)')
    ax4.set_ylabel('Dune migration rate (m/min)')

    f.subplots_adjust(hspace=0.3)
    
def make_animation(verts):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(12, 3))
    ax = plt.axes(xlim=(0, 12), ylim=(0, 0.1))

    x0 = verts[0,:,0]
    z0 = verts[0,:,1]

    plt.plot(x0,z0)
    line, = ax.plot([], [], lw=2)



    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        x = verts[i,:,0]
        y = verts[i,:,1]
        line.set_data(x, y)
        return line,
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=verts.shape[0], interval=80, blit=True)
    return anim