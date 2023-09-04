import matplotlib.pyplot as plt
import os

def generate_images(model, test_input, tar, img_result_training, epoch):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    fig = plt.figure(figsize=(10,10))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    fig.savefig(os.path.join(img_result_training, f'image{epoch}.png'), bbox_inches='tight')
    plt.close(fig)