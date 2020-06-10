
import torch
import onnx
import warnings
from trainer.utils import warn_with_traceback
warnings.showwarning = warn_with_traceback

if __name__ == '__main__':
    import argparse
    from nn import load as load_model
    from nn import create as create_model
    from nn.basenet import BASENET_CHOICES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", type=str, help='pre/trained model file')
    parser.add_argument('--arch', default='ssd300', help='model architecture', type=str)
    parser.add_argument('--basenet', default='vgg16', choices=BASENET_CHOICES, help='base net for feature extracting')
    parser.add_argument("-o", "--output", type=str, required=True, help='output file of onnx model')
    parser.add_argument("--image-size", type=int, default=1600, help='input image size')
    parser.add_argument("-v", "--verbose", action='count', default=0, help="level of debug messages")
    args = parser.parse_args()

    if args.model:
        print('Load torch model')
        model = load_model(args.model)
    else:
        print('Create torch model')
        model = create_model(arch=args.arch, classnames=['background', 'pos'], basenet=args.basenet)

    print('Trace torch model')
    # Input to the model
    batch_size = 1
    x = torch.randn(batch_size, 3, args.image_size, args.image_size, requires_grad=True)
    model.eval()
    output = model(x)
    print(output.shape)

    print('Export onnx model')
    #################### PATCH BEGIN ############################
    from torch.onnx import symbolic_helper as sym_help
    from torch.onnx import symbolic_opset11

    def clamp_min(g, self, min):
        dtype = self.type().scalarType()

        def _cast_if_not_none(tensor, dtype):
            if tensor is not None and not sym_help._is_none(tensor):
                return g.op("Cast", tensor, to_i=sym_help.cast_pytorch_to_onnx[dtype])
            else:
                return tensor

        if dtype is not None:
            min = _cast_if_not_none(min, dtype)

        return g.op("Clip", self, min)
    symbolic_opset11.clamp_min = clamp_min
    #################### PATCH END ############################
    model.trace_mode = True
    torch.onnx.export(model,                     # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      args.output,               # where to save the model (can be a file or file-like object)
                      #export_params=False,        # store the trained parameter weights inside the model file
                      #do_constant_folding=True,  # whether to execute constant folding for optimization
                      #opset_version=9,          # the ONNX version to export the model to
                      input_names=['input'],     # the model's input names
                      output_names=['output'],   # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                      #               'loc': {0: 'batch_size'},
                      #               'conf': {0: 'batch_size'}},
                      verbose=(args.verbose > 0)
                      )

    print('Check onnx model')
    onnx_model = onnx.load(args.output)
    if args.verbose > 1:
        print('The model is:\n{}'.format(onnx_model))
    onnx.checker.check_model(onnx_model)

