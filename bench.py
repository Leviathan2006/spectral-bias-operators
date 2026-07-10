import time
import torch

d = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.__version__, '| cuda', torch.version.cuda)
if d == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('arch', torch.cuda.get_arch_list())
    print('sm_120 in build:', 'sm_120' in torch.cuda.get_arch_list())


def t(fn, n=50):
    for _ in range(5):
        fn()
    if d == 'cuda':
        torch.cuda.synchronize()
    a = time.time()
    for _ in range(n):
        fn()
    if d == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - a) / n * 1e3


x = torch.randn(32, 32, 64, 64, device=d)
w = torch.randn(32, 32, 32, 32, dtype=torch.cfloat, device=d)
xf = torch.fft.rfft2(x)[:, :, :32, :32]
k = torch.randn(32, 32, 1, 1, device=d)

print('fft   %7.3f ms' % t(lambda: torch.fft.irfft2(torch.fft.rfft2(x), s=(64, 64))))
print('cmul  %7.3f ms' % t(lambda: torch.einsum('bixy,ioxy->boxy', xf, w)))
print('conv  %7.3f ms' % t(lambda: torch.nn.functional.conv2d(x, k)))
mw = torch.randn(32, 32, device=d)
print('mix   %7.3f ms' % t(lambda: torch.einsum('bihw,oi->bohw', x, mw)))   # matmul 1x1 replacement
print('gelu  %7.3f ms' % t(lambda: torch.nn.functional.gelu(x)))

# each should be well under a millisecond on a modern gpu. tens of ms => the
# fft/complex kernels in this build are the bottleneck (grab a newer cu128 wheel).
