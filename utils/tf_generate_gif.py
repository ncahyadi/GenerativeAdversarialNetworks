import tensorflow_docs.vis.embed as embed
import imageio
import os
import glob

def create_gif(id_experiment, img_result_training):
    anim_file = id_experiment+'.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
        os.chdir(img_result_training)
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    # embed.embed_file(anim_file)
