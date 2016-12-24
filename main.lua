require 'nn'
require 'image'
require 'torch'
require 'env'
require 'trepl'

local cmd = torch.CmdLine()
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-batchsize', 128, 'batchsize')
cmd:option('-epochs', 10 , 'epochs')
local config = cmd:parse(arg)

local tnt   = require 'torchnet'
local dbg   = require 'debugger'

---------------------- PREPROCESSING ----------------------
local base_data_path = "/Users/mohammadafshar1/projects/flower-recognition/data"

local network = require("./model/model.lua")
local criterion = nn.CrossEntropyCriterion()
local lr = config.lr
local epochs = config.epochs
local dataset_size = 1360
local data_dim

local all_images = torch.Tensor(dataset_size, data_dim)
local all_labels = torch.Tensor(1, data_dim)

local ndx = 1
for dir = 0, 16 do
	label_dir = './data/' .. dir
	f = io.popen('ls ' .. label_dir)
    for file in f:lines() do
		img_filename = label_dir .. '/' .. file
        img = image.load(img_filename)
        all_images[ndx] = img:view(img:nElement())
		all_labels[{{}, ndx}] = dir
        ndx = ndx + 1
    end
end


print(all_labels)
os.exit()

local labels_shuffle = torch.randperm(dataset_size)

-- create train set:
local train_data = {
   data = torch.Tensor(dataset_size, data_dim),
   labels = torch.Tensor(1, dataset_size),
   size = function() return dataset_size end
}

for i=1, dataset_size do
   train_data.data[i] = all_images[labels_shuffle[i]]:clone()
   train_data.labels[{{}, i}] = all_labels[{ {1}, {labels_shuffle[i]} }]
end

local datasets = {[1]={}, [2]={}, [3]={}, [4]={}}

datasets[1].data = torch.reshape(train_data.data[{{1, 340}}], data_dim, 340):byte()
datasets[2].data = torch.reshape(train_data.data[{{10415, 20828}}], data_dim, 340):byte()
datasets[3].data = torch.reshape(train_data.data[{{20829, 31242}}], data_dim, 340):byte()
datasets[4].data = torch.reshape(train_data.data[{{31243, dataset_size}}], data_dim, 340):byte()

datasets[1].labels = train_data.labels[{ {1}, {1, 340} }]:byte()
datasets[2].labels = train_data.labels[{ {1}, {341, 20828} }]:byte()
datasets[3].labels = train_data.labels[{ {1}, {20829, 31242} }]:byte()
datasets[4].labels = train_data.labels[{ {1}, {31243, dataset_size} }]:byte()


local function getCifarIterator(datasets)
    local listdatasets = {}
    for _, dataset in pairs(datasets) do
		-- print(_) --index
		-- print(dataset) --actual data/labels
        local list = torch.range(1, 340):totable()
        table.insert(listdatasets,
                    tnt.ListDataset{
                        list = list,
                        load = function(idx)
							-- print(idx)
							-- print(dataset.data[{{}, 251}])
                            return {
                                input  = dataset.data[{{}, idx}],
                                target = dataset.labels[{{}, idx}]
                            } -- sample contains input and target
                        end
                    })
    end
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = config.batchsize,
            dataset = tnt.ShuffleDataset{
               dataset = tnt.TransformDataset{
                    transform = function(x)
		       return {
			  input  = x.input:double():reshape(3,70,70),
            --   input  = x.input:double(),
			  target = x.target:long():add(1),
		       }
                    end,
                    dataset = tnt.ConcatDataset{
                        datasets = listdatasets
                    }
                },
            }
        }
    }
end


trainiterator = getCifarIterator(datasets)

---------------------- NETWORK ARCHITECTURE ----------------------

local network = require("./model/model.lua")
local criterion = nn.CrossEntropyCriterion()
local lr = config.lr
local epochs = config.epochs

print("Started training!")

for epoch = 1, epochs do
    local timer = torch.Timer()
    local loss = 0
    local errors = 0
    local count = 0
    for d in trainiterator() do
        network:forward(d.input)
        criterion:forward(network.output, d.target)
        network:zeroGradParameters()
        criterion:backward(network.output, d.target)
        network:backward(d.input, criterion.gradInput)
        network:updateParameters(lr)
        loss = loss + criterion.output --criterion already averages over minibatch
        count = count + 1
        local _, pred = network.output:max(2)
        errors = errors + (pred:size(1) - pred:eq(d.target):sum())
    end
    loss = loss / count
	print(string.format(
    'train | epoch = %d | lr = %1.4f | loss: %2.4f | error: %2.4f | s/iter: %2.4f',
    epoch, lr, loss, errors, timer:time().real
    ))
	torch.save("./model/history/3/config1.t7", network)
end
torch.save("./model/history/3/config1_final.t7", network)
