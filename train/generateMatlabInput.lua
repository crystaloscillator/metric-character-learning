--[[
Main Driver for Crepe
By Xiang Zhang @ New York University
]]

-- Necessary functionalities
require("nn")
require('optim')

-- Local requires
require("data")
require("model")
require("test_matlab")
require("gnuplot")
require("config")
require("xlua")

require('lfs')

-- Configurations

-- Prepare random number generator
math.randomseed(os.time())
torch.manualSeed(os.time())

-- Create namespaces
main = {}

-- The main program
function main.main()

   main.argparse()

   main.clock = {}
   main.clock.log = 0

   main.new()
   main.run()
 
end



-- Parse arguments
function main.argparse()
   local cmd = torch.CmdLine()

   -- Options
   cmd:option("-resume",0,"Resumption point in epoch. 0 means not resumption.")
   cmd:option("-debug",0,"debug. 0 means not debug.")
   cmd:option("-device",0,"device. 0 means cpu.")
   --cmd:option("-format","","stk or py")
   --cmd:option("-model","","lstm or cnn")
   cmd:text()

   -- Parse the option
   local opt = cmd:parse(arg or {})

   if opt.debug > 0 then
      dbg = require("debugger")
   end

   if opt.device > 0 then
      require("cutorch")
      require("cunn")
      cutorch.setDevice(opt.device)
      print("Device set to ".. opt.device)
      config.main.type = "torch.CudaTensor"
   else
      config.main.type = "torch.DoubleTensor"
   end

   -- Resumption operation
   if opt.resume > 0 then
      -- Find the main resumption file
      local files = main.findFiles(paths.concat(config.main.save,"main_"..tostring(opt.resume).."_*.t7b"))
      if #files ~= 1 then
    error("Found "..tostring(#files).." main resumption point.")
      end
      config.main.resume = files[1]
      print("Using main resumption point "..config.main.resume)
      -- Find the model resumption file
      local files = main.findFiles(paths.concat(config.main.save,"sequential_"..tostring(opt.resume).."_*.t7b"))
      if #files ~= 1 then
    error("Found "..tostring(#files).." model resumption point.")
      end
      config.model.file = files[1]
      print("Using model resumption point "..config.model.file)
      -- Resume the training epoch
      config.train.epoch = tonumber(opt.resume) + 1
      print("Next training epoch resumed to "..config.train.epoch)
      -- Don't do randomize
      if config.main.randomize then
         config.main.randomize = nil
         print("Disabled randomization for resumption")
      end
   end

   return opt
end

-- Train a new experiment
function main.new()
   -- Load the data
   print("Loading datasets...")
   main.train_data = Data(config.train_data)
   main.val_data = Data(config.val_data)
   
   -- Load the model
   print("Loading the model...")
   main.model = Model(config.model)
   main.model:type(config.main.type)
   print("Current model type: "..main.model:type())
   collectgarbage()

   -- Initiate the tester
   print("Loading the tester...")
   main.test_val = Test_matlab(main.train_data, main.val_data, main.model)


   collectgarbage()
end

-- Start the training
function main.run()

   if config.main.validate == true then
     print("Disabling dropouts")
     main.model:disableDropouts()
     main.test_val:run(main.testlog)
   end

   collectgarbage()
end



-- The training logging function
function main.trainlog(train)
   if config.main.collectgarbage and math.fmod(train.epoch-1,config.main.collectgarbage) == 0 then
      print("Collecting garbage at epoch = "..(train.epoch-1))
      collectgarbage()
   end

   if (os.time() - main.clock.log) >= (config.main.logtime or 1) then
      local msg = ""

	     msg = msg.."epo: "..(train.epoch-1)..
	    ", rat: "..string.format("%.2e",train.rate)..
	    ", err: "..string.format("%.2e",train.error)..
	    ", obj: "..string.format("%.2e",train.objective)

      print(msg)
   
      main.clock.log = os.time()
   end
end

function main.testlog(test)
   if config.main.collectgarbage and math.fmod(test.n,config.train_data.batch_size*config.main.collectgarbage) == 0 then
      print("Collecting garbage at n = "..test.n)
      collectgarbage()
   end
   if (os.time() - main.clock.log) >= (config.main.logtime or 1) then
      
      print("n: "..test.n..
	       ", e: "..string.format("%.2e",test.e)..
	       ", l: "..string.format("%.2e",test.l)..
	       ", err: "..string.format("%.2e",test.err)..
	       ", obj: "..string.format("%.2e",test.objective))
      main.clock.log = os.time()
   end
end

-- Utility function: find files with the specific 'ls' pattern
function main.findFiles(pattern)
   require("sys")
   local cmd = "ls "..pattern
   local str = sys.execute(cmd)
   local files = {}
   for file in str:gmatch("[^\n]+") do
      files[#files+1] = file
   end
   return files
end

-- Execute the main program
main.main()
