import torch
import torch.nn as nn
from pytorch_wavelets.dtcwt.coeffs import biort as _biort, qshift as _qshift
from pytorch_wavelets.dtcwt.lowlevel import prep_filt

from .lowlevel import mode_to_int
from .lowlevel import ScatLayerj1_f, ScatLayerj1_rot_f
from .lowlevel import ScatLayerj2_f, ScatLayerj2_rot_f


class ScatLayer(nn.Module):


    def __init__(self, biort='near_sym_a', mode='symmetric', magbias=1e-2,
                 combine_colour=False):
        super().__init__()
        self.biort = biort


        self.mode_str = mode
        self.mode = mode_to_int(mode)
        self.magbias = magbias
        self.combine_colour = combine_colour
        if biort == 'near_sym_b_bp':
            self.bandpass_diag = True
            h0o, _, h1o, _, h2o, _ = _biort(biort)
            self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
            self.h2o = torch.nn.Parameter(prep_filt(h2o, 1), False)
        else:
            self.bandpass_diag = False
            h0o, _, h1o, _ = _biort(biort)
            self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)

    def forward(self, x):


        _, ch, r, c = x.shape
        if r % 2 != 0:
            x = torch.cat((x, x[:,:,-1:]), dim=2)
        if c % 2 != 0:
            x = torch.cat((x, x[:,:,:,-1:]), dim=3)

        if self.combine_colour:
            assert ch == 3

        if self.bandpass_diag:
            Z = ScatLayerj1_rot_f.apply(
                x, self.h0o, self.h1o, self.h2o, self.mode, self.magbias,
                self.combine_colour)
        else:
            Z = ScatLayerj1_f.apply(
                x, self.h0o, self.h1o, self.mode, self.magbias,
                self.combine_colour)
        if not self.combine_colour:
            b, _, c, h, w = Z.shape
            Z = Z.view(b, 7*c, h, w)
        return Z

    def extra_repr(self):
        return "biort='{}', mode='{}', magbias={}".format(
               self.biort, self.mode_str, self.magbias)


class ScatLayerj2(nn.Module):


    def __init__(self, biort='near_sym_a', qshift='qshift_a', mode='symmetric',
                 magbias=1e-2, combine_colour=False):
        super().__init__()
        self.biort = biort
        self.qshift = biort


        self.mode_str = mode
        self.mode = mode_to_int(mode)
        self.magbias = magbias
        self.combine_colour = combine_colour
        if biort == 'near_sym_b_bp':
            assert qshift == 'qshift_b_bp'
            self.bandpass_diag = True
            h0o, _, h1o, _, h2o, _ = _biort(biort)
            self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
            self.h2o = torch.nn.Parameter(prep_filt(h2o, 1), False)
            h0a, h0b, _, _, h1a, h1b, _, _, h2a, h2b, _, _ = _qshift('qshift_b_bp')
            self.h0a = torch.nn.Parameter(prep_filt(h0a, 1), False)
            self.h0b = torch.nn.Parameter(prep_filt(h0b, 1), False)
            self.h1a = torch.nn.Parameter(prep_filt(h1a, 1), False)
            self.h1b = torch.nn.Parameter(prep_filt(h1b, 1), False)
            self.h2a = torch.nn.Parameter(prep_filt(h2a, 1), False)
            self.h2b = torch.nn.Parameter(prep_filt(h2b, 1), False)
        else:
            self.bandpass_diag = False
            h0o, _, h1o, _ = _biort(biort)
            self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
            h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
            self.h0a = torch.nn.Parameter(prep_filt(h0a, 1), False)
            self.h0b = torch.nn.Parameter(prep_filt(h0b, 1), False)
            self.h1a = torch.nn.Parameter(prep_filt(h1a, 1), False)
            self.h1b = torch.nn.Parameter(prep_filt(h1b, 1), False)

    def forward(self, x):

        ch, r, c = x.shape[1:]
        rem = r % 8
        if rem != 0:
            rows_after = (9-rem)//2
            rows_before = (8-rem) // 2
            x = torch.cat((x[:,:,:rows_before], x,
                           x[:,:,-rows_after:]), dim=2)
        rem = c % 8
        if rem != 0:
            cols_after = (9-rem)//2
            cols_before = (8-rem) // 2
            x = torch.cat((x[:,:,:,:cols_before], x,
                           x[:,:,:,-cols_after:]), dim=3)

        if self.combine_colour:
            assert ch == 3

        if self.bandpass_diag:
            pass
            Z = ScatLayerj2_rot_f.apply(
                x, self.h0o, self.h1o, self.h2o, self.h0a, self.h0b, self.h1a,
                self.h1b, self.h2a, self.h2b, self.mode, self.magbias,
                self.combine_colour)
        else:
            Z = ScatLayerj2_f.apply(
                x, self.h0o, self.h1o, self.h0a, self.h0b, self.h1a,
                self.h1b, self.mode, self.magbias, self.combine_colour)

        if not self.combine_colour:
            b, _, c, h, w = Z.shape
            Z = Z.view(b, 49*c, h, w)
        return Z

    def extra_repr(self):
        return "biort='{}', mode='{}', magbias={}".format(
               self.biort, self.mode_str, self.magbias)
