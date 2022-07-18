import torch
import torch.nn as nn
import torch.nn.functional as F

from .gma_update import GMAUpdateBlock
from .extractor import BasicEncoder
from .gma_corr import CorrBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8
from .gma import Attention, Aggregate
from .gma_network import RAFTGMA

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

class GMAL2L(RAFTGMA):
    def __init__(self, args):
        super(GMAL2L, self).__init__(args)
        self.grad_update_block = GMAUpdateBlock(self.args, hidden_dim=self.hidden_dim)

    def forward(self, image1, image2, ci1=None, ci2=None, ox=None, oy=None, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        if ci1 is not None:
            ci1 = 2 * (ci1 / 255.0) - 1.0
            ci2 = 2 * (ci2 / 255.0) - 1.0

            ci1 = ci1.contiguous()
            ci2 = ci2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            # attention, att_c, att_p = self.att(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if test_mode or itr < iters // 2:
                    net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)
                else:
                    if itr == iters // 2:
                        if ci1 is not None:
                            _, _, orig_h, orig_w = image1.shape
                            _, _, targ_h, targ_w = ci1.shape
                            ox_ = ox[0]
                            oy_ = oy[0]


                            net = torch.nn.functional.pad(net, (ox_//8, (targ_w-ox_-orig_w)//8,
                                                                        oy_//8, (targ_h-oy_-orig_h)//8))
                            flow = torch.nn.functional.pad(flow, (ox_//8, (targ_w-ox_-orig_w)//8,
                                                                        oy_//8, (targ_h-oy_-orig_h)//8))

                            coords0, _ = self.initialize_flow(ci1)
                            coords1 = flow + coords0

                            tfmap1, tfmap2 = self.fnet([ci1, ci2])
                            tfmap1 = tfmap1.float()
                            tfmap2 = tfmap2.float()
                            corr_fn = CorrBlock(tfmap1, tfmap2, radius=self.args.corr_radius)
                            corr = corr_fn(coords1)

                            cnet = self.cnet(ci1)
                            _, inp = torch.split(cnet, [hdim, cdim], dim=1)
                            inp = torch.relu(inp)
                            attention = self.att(inp)

                        net = net.detach()
                        corr = corr.detach()
                        inp = inp.detach()
                        flow = flow.detach()
                        attention = attention.detach()

                    net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            if not test_mode and itr >= iters // 2:
                flow_up = flow_up[:,:,oy_: oy_+orig_h, ox_: ox_+orig_w]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up


        return flow_predictions
