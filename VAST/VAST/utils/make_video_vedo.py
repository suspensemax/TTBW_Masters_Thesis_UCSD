from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
import numpy as np

def make_video(surface_properties_dict,num_nodes, sim):    
    surface_names = list(surface_properties_dict.keys())
    nt = num_nodes - 1 
    
    axs = Axes(
        xrange=(0, 35),
        yrange=(-10, 10),
        zrange=(-3, 0.5),
    )
    # video = Video("fixvd.gif",backend='imageio')
    video = Video("fixvd.gif",backend='ffmpeg')
    for i in range(nt - 1):
        plt = Plotter(
            bg='beige',
            bg2='lb',
            offscreen=False,
            interactive=1)
        # Any rendering loop goes here, e.g.:
        for surface_name in surface_names:
            vps = Points(np.reshape(sim[surface_name][i, :, :, :], (-1, 3)),
                        r=8,
                        c='red')
            plt += [vps, __doc__]
            # plt += __doc__
            vps = Points(np.reshape(sim['op_'+surface_name+'_wake_coords'][i, 0:i, :, :],
                                    (-1, 3)),
                        r=8,
                        c='blue')
            plt += [vps, __doc__]
            # plt += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        plt.show( elevation=-60, azimuth=-0,
                axes=False)  # render the scene
        video.add_frame()  # add individual frame
        # time.sleep(0.1)
        plt.interactive().close()
        # plt.interactive().close()
    #     plt.closeWindow()
    # plt.closeWindow()
    video.close()  # merge all the recorded frames
    
    plt.interactive().close()

