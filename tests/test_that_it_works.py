import os

import matplotlib.pyplot as plt
import torch

import visualbench as vb


def test_function_descent():
    benchmark = vb.FunctionDescent("booth")
    opt = torch.optim.Adam(benchmark.parameters(), 1e-1)
    benchmark.run(opt, 1000)
    assert benchmark.lowest_loss < 1e-4

def test_graph_layout():
    benchmark = vb.GraphLayout(vb.GraphLayout.GRID())
    opt = torch.optim.Adam(benchmark.parameters(), 10)
    benchmark.run(opt, 1000)
    assert benchmark.lowest_loss < 1800

def test_algebras_dont_error():
    benchmark = vb.MoorePenrose(vb.data.SANIC96, algebra='tropical')
    opt = torch.optim.Adam(benchmark.parameters(), 1e-1)
    benchmark.run(opt, 10)

def test_plotting_doesnt_error():
    benchmark = vb.MoorePenrose(vb.data.SANIC96)
    opt = torch.optim.Adam(benchmark.parameters(), 1e-1)
    benchmark.run(opt, 10)
    benchmark.plot()
    plt.close()

def test_rendering_doesnt_error():
    benchmark = vb.MoorePenrose(vb.data.SANIC96)
    opt = torch.optim.Adam(benchmark.parameters(), 1e-1)
    benchmark.run(opt, 10)
    benchmark.render("test.mp4")
    os.remove("test.mp4")


def test_datasets_work():
    benchmark = vb.SynthSeg1d(
        vb.models.vision.ConvNetAutoencoder(1, 1, 5, 32),
        batch_size=32,
        test_batch_size=128,
        criterion=torch.nn.functional.cross_entropy
    )
    opt = torch.optim.Adam(benchmark.parameters(), 1e-1)
    benchmark.run(opt, 10)
