def sfb1d_atrous(lo, hi, g0, g1, mode='periodization', dim=-1, dilation=1,
                 pad1=None, pad=None):


    C = lo.shape[1]
    d = dim % 4


    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L

    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)
    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)


    centre = L / 2
    fsz = (L-1)*dilation + 1
    newcentre = fsz / 2
    before = newcentre - dilation*centre


    short_offset = dilation - 1
    centre_offset = fsz % 2
    a = fsz//2
    b = fsz//2 + (fsz + 1) % 2


    pad = (0, 0, a, b) if d == 2 else (a, b, 0, 0)
    lo = mypad(lo, pad=pad, mode=mode)
    hi = mypad(hi, pad=pad, mode=mode)
    unpad = (fsz - 1, 0) if d == 2 else (0, fsz - 1)
    unpad = (0, 0)
    y = F.conv_transpose2d(lo, g0, padding=unpad, groups=C, dilation=dilation) + \
        F.conv_transpose2d(hi, g1, padding=unpad, groups=C, dilation=dilation)


    return y/(2*dilation)


def sfb2d_atrous(ll, lh, hl, hh, filts, mode='zero'):


    tensorize = [not isinstance(x, torch.Tensor) for x in filts]
    if len(filts) == 2:
        g0, g1 = filts
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(g0, g1)
        else:
            g0_col = g0
            g0_row = g0.transpose(2,3)
            g1_col = g1
            g1_row = g1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(*filts)
        else:
            g0_col, g1_col, g0_row, g1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lo = sfb1d_atrous(ll, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d_atrous(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d_atrous(lo, hi, g0_row, g1_row, mode=mode, dim=3)

    return y


class SWTInverse(nn.Module):


    def __init__(self, wave='db1', mode='zero', separable=True):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]

        if separable:
            filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
            self.register_buffer('g0_col', filts[0])
            self.register_buffer('g1_col', filts[1])
            self.register_buffer('g0_row', filts[2])
            self.register_buffer('g1_row', filts[3])
        else:
            filts = lowlevel.prep_filt_sfb2d_nonsep(
                g0_col, g1_col, g0_row, g1_row)
            self.register_buffer('h', filts)
        self.mode = mode
        self.separable = separable

    def forward(self, coeffs):


        yl, yh = coeffs
        ll = yl


        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)


            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]


            if self.separable:
                lh, hl, hh = torch.unbind(h, dim=2)
                filts = (self.g0_col, self.g1_col, self.g0_row, self.g1_row)
                ll = lowlevel.sfb2d(ll, lh, hl, hh, filts, mode=self.mode)
            else:
                c = torch.cat((ll[:,:,None], h), dim=2)
                ll = lowlevel.sfb2d_nonsep(c, self.h, mode=self.mode)
        return ll
