About
=====

**I'm archiving this project simply because there is not a robust dataset out there to build this model with. The various attempts I made at taking real world terrain data, reversing the erosion process, and training on that only led to shortcut learning and/or bad results. While there are some generative approachs you could take (diffusion models), they aren't an approach I'm interested right now (I did investigate them in a branch on this repo.) Feel free to open an issue if you have a question. Just know that the code here and the concept itself is still incomplete.**

This project aims to create a deep learning network that accelerates terrain modeling.

## Setup

To get started training a model, create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip3 install -r requirements.txt
```

### Building a Dataset

Next, you'll need to download a DEM dataset from [USGS](https://earthexplorer.usgs.gov/).
You'll want one with a lot of good erosion-like features.
For reference, TBDEMCB00225 from the CoNED TBDEM dataset is a good starting point:

<p align="center">
<img src="https://earthexplorer.usgs.gov/index/resizeimage?img=https%3A%2F%2Fims.cr.usgs.gov%2Fbrowse%2Ftopobathy%2F2015%2FTBDEMCB00225.jpg&angle=0&size=300">
</p>

Once you've got the ZIP file, extract it and place the TIFF file into `data/TBDEMCB00225`.
You can give the sub-folder a difference name if you'd like,
but the entity ID is a good way to ensure it is unique.

### Configuration

Now, you'll have to create a configuration file in order to describe how your model will be trained.
An easy way to do this is to just run the bootstrap module:

```bash
python3 -m deepslope.bootstrap
```

*VS Code users: You can also just run the `Bootstrap` launch configuration.*

Open up `config.json` and look for `dems`.
Add to this array the path of the TIFF file you extracted from the USGS download.
For example:

```json
{
    "dems": [
        "data/TBDEMCB00225/Chesapeake_Topobathy_DEM_v1_148.TIF"
    ]
}
```

The rest of the configuration fields can be left to their defaults.

### Training the Model

In order to train the model, run:

```bash
python3 -m deepslope.optim.train
```

*VS Code users: you can run the `Train Net` launch configuration.*

At the end of each epoch, a test is done and saved to `tmp` in the repo directory. You can examine this files to monitor the training progress.

## Examples

Here's a before and after render of the first prototype model.
The input was made with simplex noise.
This primarily is meant as a proof of concept, the model is still under design.

![Before](https://github.com/user-attachments/assets/56b68181-8350-4941-ad61-984ab6a306cf)
![After](https://github.com/user-attachments/assets/13137af6-3302-4e53-8994-8cc8a8e2f662)
