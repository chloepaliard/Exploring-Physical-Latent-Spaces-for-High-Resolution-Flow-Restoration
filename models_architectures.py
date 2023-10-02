##### Karman vortex street scenario #####

def karman_ato_model():
    
    with tf.name_scope('model_encoder') as scope:
        layers = [keras.layers.Input(shape=[64, 32, 3])]

        layers += [keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layers[-1])]
        layers += [keras.layers.LeakyReLU()(layers[-1])]
        layers += [keras.layers.Conv2D(filters=16, kernel_size=5, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layers[-1])]
        layers += [keras.layers.LeakyReLU()(layers[-1])]
        layers += [keras.layers.Conv2D(filters=2,  kernel_size=5, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layers[-1])]
        encoder = keras.models.Model(inputs=layers[0], outputs=layers[-1], name='encoder')

    with tf.name_scope('model_corrector') as scope:
        corr_input = keras.layers.Input(shape=[64, 32, 3])
        cblock0 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(corr_input)
        cblock0 = keras.layers.LeakyReLU()(cblock0)

        cblock1 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock0)
        cblock1 = keras.layers.LeakyReLU()(cblock1)
        cblock1 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock1)
        cblock1 = keras.layers.add([cblock0, cblock1])
        cblock1 = keras.layers.LeakyReLU()(cblock1)

        cblock2 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock1)
        cblock2 = keras.layers.LeakyReLU()(cblock2)
        cblock2 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock2)
        cblock2 = keras.layers.add([cblock1, cblock2])
        cblock2 = keras.layers.LeakyReLU()(cblock2)

        cblock3 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock2)
        cblock3 = keras.layers.LeakyReLU()(cblock3)
        cblock3 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock3)
        cblock3 = keras.layers.add([cblock2, cblock3])
        cblock3 = keras.layers.LeakyReLU()(cblock3)

        cblock4 =keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock3)
        cblock4 = keras.layers.LeakyReLU()(cblock4)
        cblock4 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock4)
        cblock4 = keras.layers.add([cblock3, cblock4])
        cblock4 = keras.layers.LeakyReLU()(cblock4)

        cblock5 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock4)
        cblock5 = keras.layers.LeakyReLU()(cblock5)
        cblock5 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock5)
        cblock5 = keras.layers.add([cblock4, cblock5])
        cblock5 = keras.layers.LeakyReLU()(cblock5)

        corr_output = keras.layers.Conv2D(filters=2, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock5)
        corrector = keras.models.Model(inputs=corr_input, outputs=corr_output, name="corrector")

    with tf.name_scope('model_decoder') as scope:
        
        dec_input = keras.layers.Input(shape=[64, 32, 2])
        upsampled = keras.layers.UpSampling2D(size=(4, 4), interpolation='nearest')(dec_input)

        with tf.name_scope('model_dsc') as scope:

            down_0 = keras.layers.MaxPooling2D((4, 4), padding='same')(upsampled)
            dsc_block_0 = keras.layers.Conv2D(filters=32, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(down_0)
            dsc_block_0 = keras.layers.LeakyReLU()(dsc_block_0)
            dsc_block_0 = keras.layers.Conv2D(filters=16, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_0)
            dsc_block_0 = keras.layers.LeakyReLU()(dsc_block_0)
            dsc_block_0 = keras.layers.UpSampling2D((2, 2))(dsc_block_0)

            down_1 = keras.layers.MaxPooling2D((2, 2), padding='same')(upsampled)
            dsc_block_1 = keras.layers.Concatenate()([dsc_block_0, down_1])
            dsc_block_1 = keras.layers.Conv2D(filters=32, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_1)
            dsc_block_1 = keras.layers.LeakyReLU()(dsc_block_1)
            dsc_block_1 = keras.layers.Conv2D(filters=16, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_1)
            dsc_block_1 = keras.layers.LeakyReLU()(dsc_block_1)
            dsc_block_1 = keras.layers.UpSampling2D((2, 2))(dsc_block_1)

            dsc_block_2 = keras.layers.Concatenate()([dsc_block_1, upsampled])
            dsc_block_2 = keras.layers.Conv2D(filters=32, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_2)
            dsc_block_2 = keras.layers.LeakyReLU()(dsc_block_2)
            dsc_block_2 = keras.layers.Conv2D(filters=16, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_2)
            dsc_block_2 = keras.layers.LeakyReLU()(dsc_block_2)

        with tf.name_scope('model_ms') as scope:

            ms_block_0 = keras.layers.Conv2D(filters=16, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)
            ms_block_0 = keras.layers.Conv2D(filters=8, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_0)
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)
            ms_block_0 = keras.layers.Conv2D(filters=8, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_0)
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)

            ms_block_1 = keras.layers.Conv2D(filters=16, kernel_size=9,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)
            ms_block_1 = keras.layers.Conv2D(filters=8, kernel_size=9,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_1)
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)
            ms_block_1 = keras.layers.Conv2D(filters=8, kernel_size=9,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_1)
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)

            ms_block_2 = keras.layers.Conv2D(filters=16, kernel_size=13,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)
            ms_block_2 = keras.layers.Conv2D(filters=8, kernel_size=13,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_2)
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)
            ms_block_2 = keras.layers.Conv2D(filters=8, kernel_size=13,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_2)
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)

            ms_block = keras.layers.Concatenate()([ms_block_0, ms_block_1, ms_block_2])
            ms_block = keras.layers.Conv2D(filters=8, kernel_size=7,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block)
            ms_block = keras.layers.LeakyReLU()(ms_block)
            ms_block = keras.layers.Conv2D(filters=3, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block)
            ms_block = keras.layers.LeakyReLU()(ms_block)

            final_block = keras.layers.Concatenate()([dsc_block_2, ms_block])

            final = keras.layers.Conv2D(filters=2, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(final_block)
            decoder = keras.models.Model(inputs=dec_input, outputs=final, name="super_res")

    return [encoder, corrector, decoder]

    
def karman_sol_model():

    with tf.name_scope('model_corrector') as scope:
        corr_input = keras.layers.Input(shape=[64, 32, 3])
        cblock0 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(corr_input)
        cblock0 = keras.layers.LeakyReLU()(cblock0)

        cblock1 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock0)
        cblock1 = keras.layers.LeakyReLU()(cblock1)
        cblock1 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock1)
        cblock1 = keras.layers.add([cblock0, cblock1])
        cblock1 = keras.layers.LeakyReLU()(cblock1)

        cblock2 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock1)
        cblock2 = keras.layers.LeakyReLU()(cblock2)
        cblock2 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock2)
        cblock2 = keras.layers.add([cblock1, cblock2])
        cblock2 = keras.layers.LeakyReLU()(cblock2)

        cblock3 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock2)
        cblock3 = keras.layers.LeakyReLU()(cblock3)
        cblock3 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock3)
        cblock3 = keras.layers.add([cblock2, cblock3])
        cblock3 = keras.layers.LeakyReLU()(cblock3)

        cblock4 =keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock3)
        cblock4 = keras.layers.LeakyReLU()(cblock4)
        cblock4 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock4)
        cblock4 = keras.layers.add([cblock3, cblock4])
        cblock4 = keras.layers.LeakyReLU()(cblock4)

        cblock5 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock4)
        cblock5 = keras.layers.LeakyReLU()(cblock5)
        cblock5 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock5)
        cblock5 = keras.layers.add([cblock4, cblock5])
        cblock5 = keras.layers.LeakyReLU()(cblock5)

        corr_output = keras.layers.Conv2D(filters=2, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock5)
  
        corrector = keras.models.Model(inputs=corr_input, outputs=corr_output, name="corrector")
  
    return [corrector]


def karman_dilresnet_model():
    
    with tf.name_scope('model_solver') as scope:
        sol_input = tf.keras.layers.Input(shape=[64, 32, 3])
        sol_input_noise = tf.keras.layers.GaussianNoise(stddev = 0.01)(sol_input, training = True)

        eblock = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding = 'same')(sol_input_noise)

        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding = 'same')(eblock)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=8, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)

        sblock2 = tf.keras.layers.add([eblock, sblock1])
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=8, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        
        sblock3 = tf.keras.layers.add([sblock1, sblock2])
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=8, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)

        sblock4 = tf.keras.layers.add([sblock2, sblock3])
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=8, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)

        decoded = tf.keras.layers.add([sblock3, sblock4])
        decoded = tf.keras.layers.Conv2D( filters=2, kernel_size=3, padding='same')(decoded)
        
        sol_output = keras.layers.add([sol_input[..., 0:2], decoded]) # [...,0:2]=velocity and [...,2:4]=force
        solver = tf.keras.models.Model(inputs=sol_input, outputs=sol_output, name="solver")

    return solver
    

def karman_sr_model():
    
    with tf.name_scope('model_decoder') as scope:
        
        dec_input = keras.layers.Input(shape=[64, 32, 2])
        upsampled = keras.layers.UpSampling2D(size=(4, 4), interpolation='nearest')(dec_input)

        with tf.name_scope('model_dsc') as scope:

            down_0 = keras.layers.MaxPooling2D((4, 4), padding='same')(upsampled)
            dsc_block_0 = keras.layers.Conv2D(filters=32, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(down_0)
            dsc_block_0 = keras.layers.LeakyReLU()(dsc_block_0)
            dsc_block_0 = keras.layers.Conv2D(filters=16, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_0)
            dsc_block_0 = keras.layers.LeakyReLU()(dsc_block_0)
            dsc_block_0 = keras.layers.UpSampling2D((2, 2))(dsc_block_0)

            down_1 = keras.layers.MaxPooling2D((2, 2), padding='same')(upsampled)
            dsc_block_1 = keras.layers.Concatenate()([dsc_block_0, down_1])
            dsc_block_1 = keras.layers.Conv2D(filters=32, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_1)
            dsc_block_1 = keras.layers.LeakyReLU()(dsc_block_1)
            dsc_block_1 = keras.layers.Conv2D(filters=16, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_1)
            dsc_block_1 = keras.layers.LeakyReLU()(dsc_block_1)
            dsc_block_1 = keras.layers.UpSampling2D((2, 2))(dsc_block_1)

            dsc_block_2 = keras.layers.Concatenate()([dsc_block_1, upsampled])
            dsc_block_2 = keras.layers.Conv2D(filters=32, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_2)
            dsc_block_2 = keras.layers.LeakyReLU()(dsc_block_2)
            dsc_block_2 = keras.layers.Conv2D(filters=16, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_2)
            dsc_block_2 = keras.layers.LeakyReLU()(dsc_block_2)

        with tf.name_scope('model_ms') as scope:

            ms_block_0 = keras.layers.Conv2D(filters=16, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)
            ms_block_0 = keras.layers.Conv2D(filters=8, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_0)
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)
            ms_block_0 = keras.layers.Conv2D(filters=8, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_0)
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)

            ms_block_1 = keras.layers.Conv2D(filters=16, kernel_size=9,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)
            ms_block_1 = keras.layers.Conv2D(filters=8, kernel_size=9,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_1)
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)
            ms_block_1 = keras.layers.Conv2D(filters=8, kernel_size=9,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_1)
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)

            ms_block_2 = keras.layers.Conv2D(filters=16, kernel_size=13,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)
            ms_block_2 = keras.layers.Conv2D(filters=8, kernel_size=13,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_2)
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)
            ms_block_2 = keras.layers.Conv2D(filters=8, kernel_size=13,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_2)
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)

            ms_block = keras.layers.Concatenate()([ms_block_0, ms_block_1, ms_block_2])
            ms_block = keras.layers.Conv2D(filters=8, kernel_size=7,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block)
            ms_block = keras.layers.LeakyReLU()(ms_block)
            ms_block = keras.layers.Conv2D(filters=3, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block)
            ms_block = keras.layers.LeakyReLU()(ms_block)

            final_block = keras.layers.Concatenate()([dsc_block_2, ms_block])

            final = keras.layers.Conv2D(filters=2, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(final_block)
            decoder = keras.models.Model(inputs=dec_input, outputs=final, name="super_res")


    return decoder


##### Forced turbulence scenario #####

# https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
def periodic_padding_flexible(tensor, axis, padding=1):
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """

    with tf.name_scope('periodic_padding') as scope:
        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(padding, int):
            padding = (padding,)

        ndim = len(tensor.shape)
        for ax,p in zip(axis,padding):
            # create a slice object that selects everything from all axes,
            # except only 0:p for the specified for right, and -p: for left

            ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
            ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
            right = tensor[ind_right]
            left = tensor[ind_left]
            middle = tensor
            tensor = tf.concat([right,middle,left], axis=ax)

        return tensor


def periodicConv2D(input, pad, **kwargs):
    ppad = periodic_padding_flexible(input, axis=(1, 2), padding=pad)
    return keras.layers.Conv2D(**kwargs)(ppad)


def forced_turb_ato_model():
    
    with tf.name_scope('model_encoder') as scope:
        layers = [keras.layers.Input(shape=[32, 32, 2])]

        layers += [periodicConv2D(layers[-1], filters=32, kernel_size=5, pad=(2,2), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))]
        layers += [keras.layers.LeakyReLU()(layers[-1])]
        layers += [periodicConv2D(layers[-1], filters=16, kernel_size=5, pad=(2,2), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))]
        layers += [keras.layers.LeakyReLU()(layers[-1])]
        layers += [periodicConv2D(layers[-1], filters=2,  kernel_size=5, pad=(2,2), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))]
        encoder = keras.models.Model(inputs=layers[0], outputs=layers[-1], name='encoder')

    with tf.name_scope('model_corrector') as scope:
        corr_input = keras.layers.Input(shape=[32, 32, 2])
        cblock0 = periodicConv2D(corr_input, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock0 = keras.layers.LeakyReLU()(cblock0)

        cblock1 = periodicConv2D(cblock0, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock1 = keras.layers.LeakyReLU()(cblock1)
        cblock1 = periodicConv2D(cblock1, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock1 = keras.layers.add([cblock0, cblock1])
        cblock1 = keras.layers.LeakyReLU()(cblock1)

        cblock2 = periodicConv2D(cblock1, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock2 = keras.layers.LeakyReLU()(cblock2)
        cblock2 = periodicConv2D(cblock2, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock2 = keras.layers.add([cblock1, cblock2])
        cblock2 = keras.layers.LeakyReLU()(cblock2)

        cblock3 = periodicConv2D(cblock2, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock3 = keras.layers.LeakyReLU()(cblock3)
        cblock3 = periodicConv2D(cblock3, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock3 = keras.layers.add([cblock2, cblock3])
        cblock3 = keras.layers.LeakyReLU()(cblock3)

        cblock4 =periodicConv2D(cblock3, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock4 = keras.layers.LeakyReLU()(cblock4)
        cblock4 = periodicConv2D(cblock4, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock4 = keras.layers.add([cblock3, cblock4])
        cblock4 = keras.layers.LeakyReLU()(cblock4)

        cblock5 = periodicConv2D(cblock4, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock5 = keras.layers.LeakyReLU()(cblock5)
        cblock5 = periodicConv2D(cblock5, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock5 = keras.layers.add([cblock4, cblock5])
        cblock5 = keras.layers.LeakyReLU()(cblock5)

        corr_output = periodicConv2D(cblock5, pad=(2,2), filters=2, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        corrector = keras.models.Model(inputs=corr_input, outputs=corr_output, name="corrector")

    with tf.name_scope('model_decoder') as scope:
        
        dec_input = keras.layers.Input(shape=[32, 32, 2])
        upsampled = keras.layers.UpSampling2D(size=(4, 4), interpolation='nearest')(dec_input)

        with tf.name_scope('model_dsc') as scope:

            down_0 = keras.layers.MaxPooling2D((4, 4), padding='valid')(upsampled)
            dsc_block_0 = periodicConv2D(down_0, pad=(1,1), filters=32, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_0 = keras.layers.LeakyReLU()(dsc_block_0)
            dsc_block_0 = periodicConv2D(dsc_block_0, pad=(1,1), filters=16, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_0 = keras.layers.LeakyReLU()(dsc_block_0)
            dsc_block_0 = keras.layers.UpSampling2D((2, 2))(dsc_block_0)

            down_1 = keras.layers.MaxPooling2D((2, 2), padding='valid')(upsampled)
            dsc_block_1 = keras.layers.Concatenate()([dsc_block_0, down_1])
            dsc_block_1 = periodicConv2D(dsc_block_1, pad=(1,1), filters=32, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_1 = keras.layers.LeakyReLU()(dsc_block_1)
            dsc_block_1 = periodicConv2D(dsc_block_1, pad=(1,1), filters=16, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_1 = keras.layers.LeakyReLU()(dsc_block_1)
            dsc_block_1 = keras.layers.UpSampling2D((2, 2))(dsc_block_1)

            dsc_block_2 = keras.layers.Concatenate()([dsc_block_1, upsampled])
            dsc_block_2 = periodicConv2D(dsc_block_2, pad=(1,1), filters=32, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_2 = keras.layers.LeakyReLU()(dsc_block_2)
            dsc_block_2 = periodicConv2D(dsc_block_2, pad=(1,1), filters=16, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_2 = keras.layers.LeakyReLU()(dsc_block_2)

        with tf.name_scope('model_ms') as scope:

            ms_block_0 = periodicConv2D(upsampled, pad=(2,2), filters=16, kernel_size=5,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)
            ms_block_0 = periodicConv2D(ms_block_0, pad=(2,2), filters=8, kernel_size=5,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)
            ms_block_0 = periodicConv2D(ms_block_0, pad=(2,2), filters=8, kernel_size=5,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)

            ms_block_1 = periodicConv2D(upsampled, pad=(4,4), filters=16, kernel_size=9,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)
            ms_block_1 = periodicConv2D(ms_block_1, pad=(4,4), filters=8, kernel_size=9,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)
            ms_block_1 = periodicConv2D(ms_block_1, pad=(4,4), filters=8, kernel_size=9,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)

            ms_block_2 = periodicConv2D(upsampled, pad=(6,6), filters=16, kernel_size=13,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)
            ms_block_2 = periodicConv2D(ms_block_2, pad=(6,6), filters=8, kernel_size=13,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)
            ms_block_2 = periodicConv2D(ms_block_2, pad=(6,6), filters=8, kernel_size=13,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)

            ms_block = keras.layers.Concatenate()([ms_block_0, ms_block_1, ms_block_2])
            ms_block = periodicConv2D(ms_block, pad=(3,3), filters=8, kernel_size=7,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block = keras.layers.LeakyReLU()(ms_block)
            ms_block = periodicConv2D(ms_block, pad=(2,2), filters=3, kernel_size=5,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block = keras.layers.LeakyReLU()(ms_block)

            final_block = keras.layers.Concatenate()([dsc_block_2, ms_block])

            final = periodicConv2D(final_block, pad=(1,1), filters=2, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            decoder = keras.models.Model(inputs=dec_input, outputs=final, name="super_res")


    return [encoder, corrector, decoder]


def forced_turb_sol_model():
    
    with tf.name_scope('model_corrector') as scope:
        corr_input = keras.layers.Input(shape=[32, 32, 2])
        cblock0 = periodicConv2D(corr_input, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock0 = keras.layers.LeakyReLU()(cblock0)

        cblock1 = periodicConv2D(cblock0, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock1 = keras.layers.LeakyReLU()(cblock1)
        cblock1 = periodicConv2D(cblock1, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock1 = keras.layers.add([cblock0, cblock1])
        cblock1 = keras.layers.LeakyReLU()(cblock1)

        cblock2 = periodicConv2D(cblock1, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock2 = keras.layers.LeakyReLU()(cblock2)
        cblock2 = periodicConv2D(cblock2, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock2 = keras.layers.add([cblock1, cblock2])
        cblock2 = keras.layers.LeakyReLU()(cblock2)

        cblock3 = periodicConv2D(cblock2, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock3 = keras.layers.LeakyReLU()(cblock3)
        cblock3 = periodicConv2D(cblock3, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock3 = keras.layers.add([cblock2, cblock3])
        cblock3 = keras.layers.LeakyReLU()(cblock3)

        cblock4 =periodicConv2D(cblock3, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock4 = keras.layers.LeakyReLU()(cblock4)
        cblock4 = periodicConv2D(cblock4, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock4 = keras.layers.add([cblock3, cblock4])
        cblock4 = keras.layers.LeakyReLU()(cblock4)

        cblock5 = periodicConv2D(cblock4, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock5 = keras.layers.LeakyReLU()(cblock5)
        cblock5 = periodicConv2D(cblock5, pad=(2,2), filters=32, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        cblock5 = keras.layers.add([cblock4, cblock5])
        cblock5 = keras.layers.LeakyReLU()(cblock5)

        corr_output = periodicConv2D(cblock5, pad=(2,2), filters=2, kernel_size=5,
        padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
        corrector = keras.models.Model(inputs=corr_input, outputs=corr_output, name="corrector")


    return corrector


def forced_turb_dilresnet_model():
    
    with tf.name_scope('model_solver') as scope:
        sol_input = tf.keras.layers.Input(shape=[32, 32, 4])
        sol_input_noise = tf.keras.layers.GaussianNoise(stddev = 0.01)(sol_input, training = True)

        eblock = periodicConv2D(sol_input_noise, pad=(1,1), filters=32, kernel_size=3, padding = 'valid')

        sblock1 = periodicConv2D(eblock, pad=(1,1), filters=32, kernel_size=3, padding = 'valid')
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = periodicConv2D(sblock1, pad = (2,2), filters=32, kernel_size=3, dilation_rate=2, padding='valid')
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = periodicConv2D(sblock1, pad = (4,4), filters=32, kernel_size=3, dilation_rate=4, padding='valid')
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = periodicConv2D(sblock1, pad = (8,8), filters=32, kernel_size=3, dilation_rate=8, padding='valid')
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = periodicConv2D(sblock1, pad = (4,4), filters=32, kernel_size=3, dilation_rate=4, padding='valid')
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = periodicConv2D(sblock1, pad = (2,2), filters=32, kernel_size=3, dilation_rate=2, padding='valid')
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = periodicConv2D(sblock1, pad = (1,1), filters=32, kernel_size=3, padding='valid')
        sblock1 = tf.keras.layers.ReLU()(sblock1)

        sblock2 = tf.keras.layers.add([eblock, sblock1])
        sblock2 = periodicConv2D(sblock2, pad = (1,1), filters=32, kernel_size=3, padding='valid')
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = periodicConv2D(sblock2, pad = (2,2), filters=32, kernel_size=3, dilation_rate=2, padding='valid')
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = periodicConv2D(sblock2, pad = (4,4), filters=32, kernel_size=3, dilation_rate=4, padding='valid')
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = periodicConv2D(sblock2, pad = (8,8), filters=32, kernel_size=3, dilation_rate=8, padding='valid')
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = periodicConv2D(sblock2, pad = (4,4), filters=32, kernel_size=3, dilation_rate=4, padding='valid')
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = periodicConv2D(sblock2, pad = (2,2), filters=32, kernel_size=3, dilation_rate=2, padding='valid')
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = periodicConv2D(sblock2, pad = (1,1), filters=32, kernel_size=3, padding='valid')
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        
        sblock3 = tf.keras.layers.add([sblock1, sblock2])
        sblock3 = periodicConv2D(sblock3, pad = (1,1), filters=32, kernel_size=3, padding='valid')
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = periodicConv2D(sblock3, pad = (2,2), filters=32, kernel_size=3, dilation_rate=2, padding='valid')
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = periodicConv2D(sblock3, pad = (4,4), filters=32, kernel_size=3, dilation_rate=4, padding='valid')
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = periodicConv2D(sblock3, pad = (8,8), filters=32, kernel_size=3, dilation_rate=8, padding='valid')
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = periodicConv2D(sblock3, pad = (4,4), filters=32, kernel_size=3, dilation_rate=4, padding='valid')
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = periodicConv2D(sblock3, pad = (2,2), filters=32, kernel_size=3, dilation_rate=2, padding='valid')
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = periodicConv2D(sblock3, pad = (1,1), filters=32, kernel_size=3, padding='valid')
        sblock3 = tf.keras.layers.ReLU()(sblock3)

        sblock4 = tf.keras.layers.add([sblock2, sblock3])
        sblock4 = periodicConv2D(sblock4, pad = (1,1), filters=32, kernel_size=3, padding='valid')
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = periodicConv2D(sblock4, pad = (2,2), filters=32, kernel_size=3, dilation_rate=2, padding='valid')
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = periodicConv2D(sblock4, pad = (4,4), filters=32, kernel_size=3, dilation_rate=4, padding='valid')
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = periodicConv2D(sblock4, pad = (8,8), filters=32, kernel_size=3, dilation_rate=8, padding='valid')
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = periodicConv2D(sblock4, pad = (4,4), filters=32, kernel_size=3, dilation_rate=4, padding='valid')
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = periodicConv2D(sblock4, pad = (2,2), filters=32, kernel_size=3, dilation_rate=2, padding='valid')
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = periodicConv2D(sblock4, pad = (1,1), filters=32, kernel_size=3, padding='valid')
        sblock4 = tf.keras.layers.ReLU()(sblock4)

        decoded = tf.keras.layers.add([sblock3, sblock4])
        decoded = periodicConv2D(decoded, pad =(1,1), filters=2, kernel_size=3, padding='valid')
        
        sol_output = keras.layers.add([sol_input[..., 0:2], decoded]) # [...,0:2]=velocity and [...,2:4]=force
        solver = tf.keras.models.Model(inputs=sol_input, outputs=sol_output, name="solver")

    return solver
    

def forced_turb_sr_model():
    
    with tf.name_scope('model_decoder') as scope:
        
        dec_input = tf.keras.layers.Input(shape=[32, 32, 2])
        upsampled = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='nearest')(dec_input)

        with tf.name_scope('model_dsc') as scope:

            down_0 = tf.keras.layers.MaxPooling2D((4, 4), padding='valid')(upsampled)
            dsc_block_0 = periodicConv2D(down_0, pad=(1,1), filters=32, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_0 = tf.keras.layers.LeakyReLU()(dsc_block_0)
            dsc_block_0 = periodicConv2D(dsc_block_0, pad=(1,1), filters=16, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_0 = tf.keras.layers.LeakyReLU()(dsc_block_0)
            dsc_block_0 = tf.keras.layers.UpSampling2D((2, 2))(dsc_block_0)

            down_1 = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(upsampled)
            dsc_block_1 = tf.keras.layers.Concatenate()([dsc_block_0, down_1])
            dsc_block_1 = periodicConv2D(dsc_block_1, pad=(1,1), filters=32, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_1 = tf.keras.layers.LeakyReLU()(dsc_block_1)
            dsc_block_1 = periodicConv2D(dsc_block_1, pad=(1,1), filters=16, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_1 = tf.keras.layers.LeakyReLU()(dsc_block_1)
            dsc_block_1 = tf.keras.layers.UpSampling2D((2, 2))(dsc_block_1)

            dsc_block_2 = tf.keras.layers.Concatenate()([dsc_block_1, upsampled])
            dsc_block_2 = periodicConv2D(dsc_block_2, pad=(1,1), filters=32, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_2 = tf.keras.layers.LeakyReLU()(dsc_block_2)
            dsc_block_2 = periodicConv2D(dsc_block_2, pad=(1,1), filters=16, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            dsc_block_2 = tf.keras.layers.LeakyReLU()(dsc_block_2)

        with tf.name_scope('model_ms') as scope:

            ms_block_0 = periodicConv2D(upsampled, pad=(2,2), filters=16, kernel_size=5,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_0 = tf.keras.layers.LeakyReLU()(ms_block_0)
            ms_block_0 = periodicConv2D(ms_block_0, pad=(2,2), filters=8, kernel_size=5,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_0 = tf.keras.layers.LeakyReLU()(ms_block_0)
            ms_block_0 = periodicConv2D(ms_block_0, pad=(2,2), filters=8, kernel_size=5,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_0 = tf.keras.layers.LeakyReLU()(ms_block_0)

            ms_block_1 = periodicConv2D(upsampled, pad=(4,4), filters=16, kernel_size=9,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_1 = tf.keras.layers.LeakyReLU()(ms_block_1)
            ms_block_1 = periodicConv2D(ms_block_1, pad=(4,4), filters=8, kernel_size=9,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_1 = tf.keras.layers.LeakyReLU()(ms_block_1)
            ms_block_1 = periodicConv2D(ms_block_1, pad=(4,4), filters=8, kernel_size=9,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_1 = tf.keras.layers.LeakyReLU()(ms_block_1)

            ms_block_2 = periodicConv2D(upsampled, pad=(6,6), filters=16, kernel_size=13,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_2 = tf.keras.layers.LeakyReLU()(ms_block_2)
            ms_block_2 = periodicConv2D(ms_block_2, pad=(6,6), filters=8, kernel_size=13,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_2 = tf.keras.layers.LeakyReLU()(ms_block_2)
            ms_block_2 = periodicConv2D(ms_block_2, pad=(6,6), filters=8, kernel_size=13,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block_2 = tf.keras.layers.LeakyReLU()(ms_block_2)

            ms_block = tf.keras.layers.Concatenate()([ms_block_0, ms_block_1, ms_block_2])
            ms_block = periodicConv2D(ms_block, pad=(3,3), filters=8, kernel_size=7,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block = tf.keras.layers.LeakyReLU()(ms_block)
            ms_block = periodicConv2D(ms_block, pad=(2,2), filters=3, kernel_size=5,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            ms_block = tf.keras.layers.LeakyReLU()(ms_block)

            final_block = tf.keras.layers.Concatenate()([dsc_block_2, ms_block])

            final = periodicConv2D(final_block, pad=(1,1), filters=2, kernel_size=3,
            padding = 'valid', kernel_regularizer=keras.regularizers.l2(0.01))
            decoder = tf.keras.models.Model(inputs=dec_input, outputs=final, name="super_res")


    return decoder



##### Smoke plume scenario #####


def smoke_plume_ato_model():
    
    with tf.name_scope('model_encoder') as scope:
        layers = [keras.layers.Input(shape=[32, 32, 3])]

        layers += [keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layers[-1])]
        layers += [keras.layers.LeakyReLU()(layers[-1])]
        layers += [keras.layers.Conv2D(filters=16, kernel_size=5, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layers[-1])]
        layers += [keras.layers.LeakyReLU()(layers[-1])]
        layers += [keras.layers.Conv2D(filters=2,  kernel_size=5, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layers[-1])]

        encoder = keras.models.Model(inputs=layers[0], outputs=layers[-1], name='encoder')

    with tf.name_scope('model_adjustment') as scope:
        corr_input = keras.layers.Input(shape=[32, 32, 3])
        cblock0 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(corr_input)
        cblock0 = keras.layers.LeakyReLU()(cblock0)

        cblock1 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock0)
        cblock1 = keras.layers.LeakyReLU()(cblock1)
        cblock1 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock1)
        cblock1 = keras.layers.add([cblock0, cblock1])
        cblock1 = keras.layers.LeakyReLU()(cblock1)

        cblock2 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock1)
        cblock2 = keras.layers.LeakyReLU()(cblock2)
        cblock2 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock2)
        cblock2 = keras.layers.add([cblock1, cblock2])
        cblock2 = keras.layers.LeakyReLU()(cblock2)

        cblock3 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock2)
        cblock3 = keras.layers.LeakyReLU()(cblock3)
        cblock3 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock3)
        cblock3 = keras.layers.add([cblock2, cblock3])
        cblock3 = keras.layers.LeakyReLU()(cblock3)

        cblock4 =keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock3)
        cblock4 = keras.layers.LeakyReLU()(cblock4)
        cblock4 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock4)
        cblock4 = keras.layers.add([cblock3, cblock4])
        cblock4 = keras.layers.LeakyReLU()(cblock4)

        cblock5 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock4)
        cblock5 = keras.layers.LeakyReLU()(cblock5)
        cblock5 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock5)
        cblock5 = keras.layers.add([cblock4, cblock5])
        cblock5 = keras.layers.LeakyReLU()(cblock5)

        corr_output = keras.layers.Conv2D(filters=2, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock5)

        adjustment = keras.models.Model(inputs=corr_input, outputs=corr_output, name="adjustment")

    with tf.name_scope('model_decoder') as scope:
        
        dec_input = keras.layers.Input(shape=[32, 32, 2])
        upsampled = keras.layers.UpSampling2D(size=(4, 4), interpolation='nearest')(dec_input)

        with tf.name_scope('model_dsc') as scope:

            down_0 = keras.layers.MaxPooling2D((4, 4), padding='same')(upsampled)
            dsc_block_0 = keras.layers.Conv2D(filters=32, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(down_0)
            dsc_block_0 = keras.layers.LeakyReLU()(dsc_block_0)
            dsc_block_0 = keras.layers.Conv2D(filters=16, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_0)
            dsc_block_0 = keras.layers.LeakyReLU()(dsc_block_0)
            dsc_block_0 = keras.layers.UpSampling2D((2, 2))(dsc_block_0)

            down_1 = keras.layers.MaxPooling2D((2, 2), padding='same')(upsampled)
            dsc_block_1 = keras.layers.Concatenate()([dsc_block_0, down_1])
            dsc_block_1 = keras.layers.Conv2D(filters=32, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_1)
            dsc_block_1 = keras.layers.LeakyReLU()(dsc_block_1)
            dsc_block_1 = keras.layers.Conv2D(filters=16, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_1)
            dsc_block_1 = keras.layers.LeakyReLU()(dsc_block_1)
            dsc_block_1 = keras.layers.UpSampling2D((2, 2))(dsc_block_1)

            dsc_block_2 = keras.layers.Concatenate()([dsc_block_1, upsampled])
            dsc_block_2 = keras.layers.Conv2D(filters=32, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_2)
            dsc_block_2 = keras.layers.LeakyReLU()(dsc_block_2)
            dsc_block_2 = keras.layers.Conv2D(filters=16, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_2)
            dsc_block_2 = keras.layers.LeakyReLU()(dsc_block_2)

        with tf.name_scope('model_ms') as scope:

            ms_block_0 = keras.layers.Conv2D(filters=16, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)
            ms_block_0 = keras.layers.Conv2D(filters=8, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_0)
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)
            ms_block_0 = keras.layers.Conv2D(filters=8, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_0)
            ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)

            ms_block_1 = keras.layers.Conv2D(filters=16, kernel_size=9,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)
            ms_block_1 = keras.layers.Conv2D(filters=8, kernel_size=9,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_1)
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)
            ms_block_1 = keras.layers.Conv2D(filters=8, kernel_size=9,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_1)
            ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)

            ms_block_2 = keras.layers.Conv2D(filters=16, kernel_size=13,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)
            ms_block_2 = keras.layers.Conv2D(filters=8, kernel_size=13,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_2)
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)
            ms_block_2 = keras.layers.Conv2D(filters=8, kernel_size=13,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_2)
            ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)

            ms_block = keras.layers.Concatenate()([ms_block_0, ms_block_1, ms_block_2])
            ms_block = keras.layers.Conv2D(filters=8, kernel_size=7,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block)
            ms_block = keras.layers.LeakyReLU()(ms_block)
            ms_block = keras.layers.Conv2D(filters=3, kernel_size=5,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block)
            ms_block = keras.layers.LeakyReLU()(ms_block)

            final_block = keras.layers.Concatenate()([dsc_block_2, ms_block])
            final_block = keras.layers.Conv2D(filters=2, kernel_size=3,
            padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(final_block)

            decoder = keras.models.Model(inputs=dec_input, outputs=final_block, name="fukami")

    return [encoder, adjustment, decoder]

def smoke_plume_sol_model():
    
    with tf.name_scope('model_corrector') as scope:
        corr_input = keras.layers.Input(shape=[32, 32, 3])
        cblock0 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(corr_input)
        cblock0 = keras.layers.LeakyReLU()(cblock0)

        cblock1 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock0)
        cblock1 = keras.layers.LeakyReLU()(cblock1)
        cblock1 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock1)
        cblock1 = keras.layers.add([cblock0, cblock1])
        cblock1 = keras.layers.LeakyReLU()(cblock1)

        cblock2 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock1)
        cblock2 = keras.layers.LeakyReLU()(cblock2)
        cblock2 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock2)
        cblock2 = keras.layers.add([cblock1, cblock2])
        cblock2 = keras.layers.LeakyReLU()(cblock2)

        cblock3 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock2)
        cblock3 = keras.layers.LeakyReLU()(cblock3)
        cblock3 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock3)
        cblock3 = keras.layers.add([cblock2, cblock3])
        cblock3 = keras.layers.LeakyReLU()(cblock3)

        cblock4 =keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock3)
        cblock4 = keras.layers.LeakyReLU()(cblock4)
        cblock4 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock4)
        cblock4 = keras.layers.add([cblock3, cblock4])
        cblock4 = keras.layers.LeakyReLU()(cblock4)

        cblock5 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock4)
        cblock5 = keras.layers.LeakyReLU()(cblock5)
        cblock5 = keras.layers.Conv2D(filters=32, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock5)
        cblock5 = keras.layers.add([cblock4, cblock5])
        cblock5 = keras.layers.LeakyReLU()(cblock5)

        corr_output = keras.layers.Conv2D(filters=2, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(cblock5)
        corrector = keras.models.Model(inputs=corr_input, outputs=corr_output, name="corrector")

    return corrector

def smoke_plume_dilresnet_model(stddev_noise):
    
    with tf.name_scope('model_solver') as scope:
        sol_input = tf.keras.layers.Input(shape=[32, 32, 3])
        sol_input_noise = tf.keras.layers.GaussianNoise(stddev = stddev_noise)(sol_input, training = True)

        eblock = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding = 'same')(sol_input_noise)

        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding = 'same')(eblock)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=8, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)
        sblock1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock1)
        sblock1 = tf.keras.layers.ReLU()(sblock1)

        sblock2 = tf.keras.layers.add([eblock, sblock1])
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=8, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        sblock2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock2)
        sblock2 = tf.keras.layers.ReLU()(sblock2)
        
        sblock3 = tf.keras.layers.add([sblock1, sblock2])
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=8, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)
        sblock3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock3)
        sblock3 = tf.keras.layers.ReLU()(sblock3)

        sblock4 = tf.keras.layers.add([sblock2, sblock3])
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=8, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=4, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=2, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)
        sblock4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(sblock4)
        sblock4 = tf.keras.layers.ReLU()(sblock4)

        sblock5 = tf.keras.layers.add([sblock3, sblock4])
        sblock5 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(sblock5)
        
        solved = keras.layers.add([sol_input, sblock5])

        clipped_density = keras.backend.clip(solved[..., 0:1], 0, 1.0)
        sol_output = keras.layers.Concatenate(axis=-1)([clipped_density, solved[..., 1:3]])

        solver = tf.keras.models.Model(inputs=sol_input, outputs=sol_output, name="solver")

    return solver

def smoke_plume_sr_model():

    with tf.name_scope('model_decoder') as scope:
        
        dec_input = keras.layers.Input(shape=[32, 32, 2])
        upsampled = keras.layers.UpSampling2D(size=(4, 4), interpolation='nearest')(dec_input)

        down_0 = keras.layers.MaxPooling2D((4, 4), padding='same')(upsampled)
        dsc_block_0 = keras.layers.Conv2D(filters=32, kernel_size=3,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(down_0)
        dsc_block_0 = keras.layers.LeakyReLU()(dsc_block_0)
        dsc_block_0 = keras.layers.Conv2D(filters=16, kernel_size=3,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_0)
        dsc_block_0 = keras.layers.LeakyReLU()(dsc_block_0)
        dsc_block_0 = keras.layers.UpSampling2D((2, 2))(dsc_block_0)

        down_1 = keras.layers.MaxPooling2D((2, 2), padding='same')(upsampled)
        dsc_block_1 = keras.layers.Concatenate()([dsc_block_0, down_1])
        dsc_block_1 = keras.layers.Conv2D(filters=32, kernel_size=3,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_1)
        dsc_block_1 = keras.layers.LeakyReLU()(dsc_block_1)
        dsc_block_1 = keras.layers.Conv2D(filters=16, kernel_size=3,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_1)
        dsc_block_1 = keras.layers.LeakyReLU()(dsc_block_1)
        dsc_block_1 = keras.layers.UpSampling2D((2, 2))(dsc_block_1)

        dsc_block_2 = keras.layers.Concatenate()([dsc_block_1, upsampled])
        dsc_block_2 = keras.layers.Conv2D(filters=32, kernel_size=3,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_2)
        dsc_block_2 = keras.layers.LeakyReLU()(dsc_block_2)
        dsc_block_2 = keras.layers.Conv2D(filters=16, kernel_size=3,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(dsc_block_2)
        dsc_block_2 = keras.layers.LeakyReLU()(dsc_block_2)

    with tf.name_scope('model_ms') as scope:

        ms_block_0 = keras.layers.Conv2D(filters=16, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
        ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)
        ms_block_0 = keras.layers.Conv2D(filters=8, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_0)
        ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)
        ms_block_0 = keras.layers.Conv2D(filters=8, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_0)
        ms_block_0 = keras.layers.LeakyReLU()(ms_block_0)

        ms_block_1 = keras.layers.Conv2D(filters=16, kernel_size=9,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
        ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)
        ms_block_1 = keras.layers.Conv2D(filters=8, kernel_size=9,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_1)
        ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)
        ms_block_1 = keras.layers.Conv2D(filters=8, kernel_size=9,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_1)
        ms_block_1 = keras.layers.LeakyReLU()(ms_block_1)

        ms_block_2 = keras.layers.Conv2D(filters=16, kernel_size=13,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(upsampled)
        ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)
        ms_block_2 = keras.layers.Conv2D(filters=8, kernel_size=13,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_2)
        ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)
        ms_block_2 = keras.layers.Conv2D(filters=8, kernel_size=13,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block_2)
        ms_block_2 = keras.layers.LeakyReLU()(ms_block_2)

        ms_block = keras.layers.Concatenate()([ms_block_0, ms_block_1, ms_block_2])
        ms_block = keras.layers.Conv2D(filters=8, kernel_size=7,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block)
        ms_block = keras.layers.LeakyReLU()(ms_block)
        ms_block = keras.layers.Conv2D(filters=3, kernel_size=5,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(ms_block)
        ms_block = keras.layers.LeakyReLU()(ms_block)

        final_block = keras.layers.Concatenate()([dsc_block_2, ms_block])
        dec_output = keras.layers.Conv2D(filters=2, kernel_size=3,
        padding = 'same', kernel_regularizer=keras.regularizers.l2(0.01))(final_block)

        decoder = keras.models.Model(inputs=dec_input, outputs=dec_output, name="fukami")

    return decoder
