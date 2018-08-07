import tensorflow as tf
import sys, os, os.path

# Data parameters
shuffle = False
input_dtype=tf.uint8
dtype=tf.float32

input_h = input_w = 1024
input_ch = 4
patch_h = patch_w = 32
image_patch_ratio = patch_h * patch_w / (input_h * input_w)
input_noise = 0

usage = "Usage: image_test.py <path to input list> <path to outputs>\n"
usage+= "where <path to input list> is a list of images to process separated by newlines\n"
usage+= "and <path to outputs> is the directory to store saved images."

def read_files(image_list):
    filename_queue = tf.train.string_input_producer(image_list, shuffle=shuffle)

    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)
    image = tf.image.decode_png(image_file, channels=input_ch, dtype=input_dtype)
    image = tf.image.convert_image_dtype(image, dtype)
    image.set_shape((input_h, input_w, input_ch))

    return image

def add_noise(image, mean=0.0, stddev=0.5):
    noise = tf.random_normal(shape=image.shape,
              mean=0.0, stddev=stddev,
              dtype=dtype)

    return image + noise

def generate_patches(image):
    patch_size = [1, patch_h, patch_w, 1]
    patches = tf.extract_image_patches([image],
        patch_size, patch_size, [1, 1, 1, 1], 'VALID')
    patches = tf.reshape(patches, [-1, patch_h, patch_w, input_ch])

    return patches

def make_image(data):
    converted = tf.image.convert_image_dtype(data, input_dtype)
    encoded = tf.image.encode_png(converted)

    return encoded

def make_images(data):
    data_queue = tf.train.batch([data],
            batch_size=1,
            enqueue_many=True,
            capacity=10000)

    return make_image(data_queue[0])

def reconstruct_image(patches):
    image = tf.reshape(patches, [1, input_h, input_w, input_ch])

    return make_image(image[0])

def main(args):
    if len(args) != 2:
        print(usage)
        sys.exit(1)

    input_list = args[0]
    image_dir = args[1]

    with open(input_list, 'r') as input_set:
        inputs = input_set.read().splitlines()

    n_examples = len(inputs)
    n_patches = n_examples // image_patch_ratio
    
    # Load, patch and reconstruct images
    input_data = read_files(inputs)
    input_img = make_image(input_data)
    input_patches = generate_patches(input_data)
    patch_imgs = make_images(input_patches)
    output_img = reconstruct_image(input_patches)

    # Initialize session and graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Main loop
        try:
            i = 0
            while not coord.should_stop():
                print("Saving image %d/%d" % (i, n_patches))
                tag = str(i)

                # Generate files for input, patches, and output
                input_name = tf.constant(os.path.join(image_dir, tag + '_in.png'))
                patch_name = tf.constant(os.path.join(image_dir, tag + '_patch.png'))
                output_name = tf.constant(os.path.join(image_dir, tag + '_out.png'))

                input_fwrite = tf.write_file(input_name, input_img)
                patch_fwrite = tf.write_file(patch_name, patch_imgs)
                output_fwrite = tf.write_file(output_name, output_img)

                # Run only patch_fwrite if you want to quickly save lots of patches
                # The input and output images will be the same every time, so don't
                # waste your breath.
                sess.run([input_fwrite, patch_fwrite, output_fwrite])

                i += 1

        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)