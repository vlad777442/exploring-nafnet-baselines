import ast
from pathlib import Path
import unittest

import torch
from torch import nn

from blocks import CBAM, NAFBlock_A5


class A5Tests(unittest.TestCase):
    def test_shapes_and_gradients(self):
        # Include narrow channels to exercise the minimum MLP width.
        for channels, height, width in ((1, 8, 9), (8, 16, 16), (32, 32, 32)):
            with self.subTest(channels=channels):
                block = NAFBlock_A5(channels)
                x = torch.randn(2, channels, height, width, requires_grad=True)
                for training in (True, False):
                    block.train(training)
                    self.assertEqual(block(x).shape, x.shape)
                block(x).square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, parameter in block.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_residual_identity_when_branch_is_zero(self):
        block = NAFBlock_A5(8)
        nn.init.zeros_(block.pw2.weight)
        nn.init.zeros_(block.pw2.bias)
        x = torch.randn(2, 8, 12, 12)
        torch.testing.assert_close(block(x), x, rtol=0, atol=0)

    def test_cbam_applies_both_attention_stages(self):
        attention = CBAM(8)
        for parameter in attention.parameters():
            nn.init.zeros_(parameter)
        x = torch.randn(2, 8, 12, 12)
        # Zero logits produce 0.5 channel and 0.5 spatial gates.
        torch.testing.assert_close(attention(x), x * 0.25)

    def test_runner_selects_a5_and_builds_network(self):
        # Execute the runner's real block import, model class, and registry
        # without loading CIFAR or initializing the LPIPS weight download.
        source = Path(__file__).resolve().parents[1] / 'train_eval_runner.py'
        tree = ast.parse(source.read_text())
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'blocks':
                nodes.append(node)
            elif isinstance(node, ast.ClassDef) and node.name == 'SimpleNAFNet':
                nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name == 'main':
                nodes.extend(
                    statement for statement in node.body
                    if isinstance(statement, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'block_map'
                            for target in statement.targets)
                )
        namespace = {'torch': torch, 'nn': nn}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        self.assertIs(namespace['block_map']['a5'], NAFBlock_A5)
        model = namespace['SimpleNAFNet'](namespace['block_map']['a5'])
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output >= 0) & (output <= 1)).all())
        output.mean().backward()


if __name__ == '__main__':
    unittest.main()
