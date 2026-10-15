import torch
import torch.nn as nn
from numpy import ndarray, sqrt

from pytorch_wavelets.dtcwt.coeffs import qshift as _qshift, biort as _biort, level1
from pytorch_wavelets.dtcwt.lowlevel import prep_filt
from pytorch_wavelets.dtcwt.transform_funcs import FWD_J1, FWD_J2PLUS
from pytorch_wavelets.dtcwt.transform_funcs import INV_J1, INV_J2PLUS
from pytorch_wavelets.dtcwt.transform_funcs import get_dimensions6
from pytorch_wavelets.dwt.lowlevel import mode_to_int
from pytorch_wavelets.dwt.transform2d import DWTForward, DWTInverse


def pm(a, b):
    u = (a + b)/sqrt(2)
    v = (a - b)/sqrt(2)
    return u, v


class DTCWTForward(nn.Module):


    def __init__(self, biort='near_sym_a', qshift='qshift_a',
                 J=3, skip_hps=False, include_scale=False,
                 o_dim=2, ri_dim=-1, mode='symmetric'):
        super().__init__()
        if o_dim == ri_dim:
            raise ValueError("Orientations and real/imaginary parts must be "
                             "in different dimensions.")

        self.biort = biort
        self.qshift = qshift
        self.J = J
        self.o_dim = o_dim
        self.ri_dim = ri_dim
        self.mode = mode
        if isinstance(biort, str):
            h0o, _, h1o, _ = _biort(biort)
            self.register_buffer('h0o', prep_filt(h0o, 1))
            self.register_buffer('h1o', prep_filt(h1o, 1))
        else:
            self.register_buffer('h0o', prep_filt(biort[0], 1))
            self.register_buffer('h1o', prep_filt(biort[1], 1))
        if isinstance(qshift, str):
            h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
            self.register_buffer('h0a', prep_filt(h0a, 1))
            self.register_buffer('h0b', prep_filt(h0b, 1))
            self.register_buffer('h1a', prep_filt(h1a, 1))
            self.register_buffer('h1b', prep_filt(h1b, 1))
        else:
            self.register_buffer('h0a', prep_filt(qshift[0], 1))
            self.register_buffer('h0b', prep_filt(qshift[1], 1))
            self.register_buffer('h1a', prep_filt(qshift[2], 1))
            self.register_buffer('h1b', prep_filt(qshift[3], 1))


        if isinstance(skip_hps, (list, tuple, ndarray)):
            self.skip_hps = skip_hps
        else:
            self.skip_hps = [skip_hps,] * self.J
        if isinstance(include_scale, (list, tuple, ndarray)):
            self.include_scale = include_scale
        else:
            self.include_scale = [include_scale,] * self.J

    def forward(self, x):


        scales = [x.new_zeros([]),] * self.J
        highs = [x.new_zeros([]),] * self.J
        mode = mode_to_int(self.mode)
        if self.J == 0:
            return x, None


        r, c = x.shape[2:]
        if r % 2 != 0:
            x = torch.cat((x, x[:,:,-1:]), dim=2)
        if c % 2 != 0:
            x = torch.cat((x, x[:,:,:,-1:]), dim=3)


        low, h = FWD_J1.apply(x, self.h0o, self.h1o, self.skip_hps[0],
                              self.o_dim, self.ri_dim, mode)
        highs[0] = h
        if self.include_scale[0]:
            scales[0] = low

        for j in range(1, self.J):

            r, c = low.shape[2:]
            if r % 4 != 0:
                low = torch.cat((low[:,:,0:1], low, low[:,:,-1:]), dim=2)
            if c % 4 != 0:
                low = torch.cat((low[:,:,:,0:1], low, low[:,:,:,-1:]), dim=3)

            low, h = FWD_J2PLUS.apply(low, self.h0a, self.h1a, self.h0b,
                                      self.h1b, self.skip_hps[j], self.o_dim,
                                      self.ri_dim, mode)
            highs[j] = h
            if self.include_scale[j]:
                scales[j] = low

        if True in self.include_scale:
            return scales, highs
        else:
            return low, highs


class DTCWTInverse(nn.Module):


    def __init__(self, biort='near_sym_a', qshift='qshift_a', o_dim=2,
                 ri_dim=-1, mode='symmetric'):
        super().__init__()
        self.biort = biort
        self.qshift = qshift
        self.o_dim = o_dim
        self.ri_dim = ri_dim
        self.mode = mode
        if isinstance(biort, str):
            _, g0o, _, g1o = _biort(biort)
            self.register_buffer('g0o', prep_filt(g0o, 1))
            self.register_buffer('g1o', prep_filt(g1o, 1))
        else:
            self.register_buffer('g0o', prep_filt(biort[0], 1))
            self.register_buffer('g1o', prep_filt(biort[1], 1))
        if isinstance(qshift, str):
            _, _, g0a, g0b, _, _, g1a, g1b = _qshift(qshift)
            self.register_buffer('g0a', prep_filt(g0a, 1))
            self.register_buffer('g0b', prep_filt(g0b, 1))
            self.register_buffer('g1a', prep_filt(g1a, 1))
            self.register_buffer('g1b', prep_filt(g1b, 1))
        else:
            self.register_buffer('g0a', prep_filt(qshift[0], 1))
            self.register_buffer('g0b', prep_filt(qshift[1], 1))
            self.register_buffer('g1a', prep_filt(qshift[2], 1))
            self.register_buffer('g1b', prep_filt(qshift[3], 1))

    def forward(self, coeffs):


        low, highs = coeffs
        J = len(highs)
        mode = mode_to_int(self.mode)
        _, _, h_dim, w_dim = get_dimensions6(
            self.o_dim, self.ri_dim)
        for j, s in zip(range(J-1, 0, -1), highs[1:][::-1]):
            if s is not None and s.shape != torch.Size([]):
                assert s.shape[self.o_dim] == 6, "Inverse transform must " \
                    "have input with 6 orientations"
                assert len(s.shape) == 6, "Bandpass inputs must have " \
                    "6 dimensions"
                assert s.shape[self.ri_dim] == 2, "Inputs must be complex " \
                    "with real and imaginary parts in the ri dimension"

                r, c = low.shape[2:]
                r1, c1 = s.shape[h_dim], s.shape[w_dim]
                if r != r1 * 2:
                    low = low[:,:,1:-1]
                if c != c1 * 2:
                    low = low[:,:,:,1:-1]

            low = INV_J2PLUS.apply(low, s, self.g0a, self.g1a, self.g0b,
                                   self.g1b, self.o_dim, self.ri_dim, mode)


        if highs[0] is not None and highs[0].shape != torch.Size([]):
            r, c = low.shape[2:]
            r1, c1 = highs[0].shape[h_dim], highs[0].shape[w_dim]
            if r != r1 * 2:
                low = low[:,:,1:-1]
            if c != c1 * 2:
                low = low[:,:,:,1:-1]

        low = INV_J1.apply(low, highs[0], self.g0o, self.g1o, self.o_dim,
                           self.ri_dim, mode)
        return low
