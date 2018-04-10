import tensorflow as tf

from DANN import DANN


def main():
    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = DANN(sess)
        model.build_model()
        # you can choose source, target or dann.
        print('\nDomain adaptation training')
        source_acc, target_acc, d_acc, dann_emb, _ = model.train_and_evaluate('dann')
        print('Source (MNIST) accuracy:', source_acc)
        print('Target (MNIST-M) accuracy:', target_acc)
        print('Domain accuracy:', d_acc)


if __name__ == '__main__':
    main()
