from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator, MultipleLocator
import scipy.stats as stats
import numpy as np
#z-score scale converter for plot
#matplotlib doesn't come with z-score scale
class PPFScale(mscale.ScaleBase):
    name = 'ppf'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis)

    def get_transform(self):
        return self.PPFTransform()

    def set_default_locators_and_formatters(self, axis):
        class VarFormatter(Formatter):
            def __call__(self, x, pos=None):
                return f'{x}'[1:]
        axis.set_major_locator(FixedLocator(np.array([0.00001, 0.000025, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99, 1.00])))
        axis.set_major_formatter(VarFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, 1e-6), min(vmax, 1-1e-6)

    class PPFTransform(mtransforms.Transform):
        input_dims = output_dims = 1 

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a): 
            return stats.norm.ppf(a)

        def inverted(self):
            return PPFScale.IPPFTransform()

    class IPPFTransform(mtransforms.Transform):
        input_dims = output_dims = 1 

        def transform_non_affine(self, a): 
            return stats.norm.cdf(a)

        def inverted(self):
            return PPFScale.PPFTransform()
#mscale.register_scale(PPFScale)

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value