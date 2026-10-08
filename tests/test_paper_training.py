"""Regression checks for public RBF prediction and volume-analysis compatibility."""
from pathlib import Path
import sys
import unittest
import numpy as np
import gpytorch
from utils.paper_training import Data, fit_fold, predict
from lib.physics import SUPPORTED_ELEMENTS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from experimental_14_anchors import head_dv, head_sigma
from borg_experiments import Data as AnalysisData, BETA
from calibration import score_block


class PaperTrainingTests(unittest.TestCase):
    def test_current_anchor_and_order_match_training_basis(self):
        data=AnalysisData()
        self.assertEqual(data.basis,list(SUPPORTED_ELEMENTS))
        self.assertEqual(BETA,.05)
        self.assertAlmostEqual(data.v_base()[data.ti].item(),3.26**3/2)
        np.testing.assert_allclose(data.comp.sum(axis=1),1.)

    def test_public_engine_supports_predictions_and_centered_volume_diagnostics(self):
        data=Data();train=np.arange(20);test=np.arange(20,25)
        fit=fit_fold(data.Xk[train],data.phys[train],data.comp[train],data.y[train],data.v_base('mlip'),n1=2,n2=2)
        self.assertIsInstance(fit['model'].covar_module.base_kernel,gpytorch.kernels.RBFKernel)
        mu,sd=predict(fit,data.Xk[test],data.phys[test],data.comp[test])
        self.assertTrue(np.isfinite(mu).all());self.assertTrue((sd>0).all())
        dv=head_dv(fit,data.comp[test]);sigma=head_sigma(fit,data.comp[test])
        np.testing.assert_allclose((data.comp[test]*dv).sum(axis=1),0.,atol=1e-12)
        np.testing.assert_allclose(sigma,(data.comp[test]*dv**2).sum(axis=1))
        base=data.v_base('mlip').numpy()
        np.testing.assert_allclose(dv,base[None,:]-(data.comp[test]@base)[:,None],atol=1e-12)

    def test_gaussian_scoring_uses_central_intervals(self):
        scores=score_block(np.array([0.,1.,3.]),np.zeros(3),np.ones(3))
        self.assertAlmostEqual(scores['cov68'],100/3)
        self.assertAlmostEqual(scores['cov95'],200/3)
        self.assertTrue(np.isfinite(list(scores.values())).all())

if __name__=='__main__':unittest.main()
