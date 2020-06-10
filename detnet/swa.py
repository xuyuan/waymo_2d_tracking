if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from trainer.swa import swa
    import nn

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, help='input directory which contains models')
    parser.add_argument("-o", "--output", type=str, default='swa_model.pth', help='output model file')
    parser.add_argument('--device', default='auto', choices=['cuda', 'cpu'], help='running with cpu or cuda')

    args = parser.parse_args()

    net = swa(nn.load, args.input, args.output, args.device)
