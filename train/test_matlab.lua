--[[ Tester for Crepe
By Xiang Zhang @ New York University
--]]

require("sys")

local Test_matlab = torch.class("Test_matlab")

function Test_matlab:__init(train_data, test_data, model, config)

   -- Store the objects
   self.train_data = train_data
   self.test_data = test_data

   self.model = model

   self.new_model = nn.Sequential()
   for i = 1,#model.sequential.modules do
      if i > 17 then
         break
      end
      local m = Model:makeCleanModule(model.sequential.modules[i])
      self.new_model:add(m)
   end
   self.new_model:type(self.model:type())

end

-- Execute testing for a batch step
function Test_matlab:run(logfunc)

   train_inputs = torch.Tensor(self.train_data.data.size, 1024):type(self.model:type())
   train_labels = torch.Tensor(train_inputs:size(1)):type(self.model:type())

   test_inputs = torch.Tensor(self.test_data.data.size, 1024):type(self.model:type())
   test_labels = torch.Tensor(test_inputs:size(1)):type(self.model:type())

   k=1
   for batch,labels,n in self.train_data:iterator() do
      self.batch = self.batch or batch:transpose(2,3):contiguous():type(self.model:type())

      self.batch:copy(batch:transpose(2, 3):contiguous())

      if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
      self.output = self.new_model:forward(self.batch)

      for i=1, self.output:size()[1] do
         if k > self.train_data.data.size then
            break
         end
         train_inputs[k] = self.output[i]
         train_labels[k] = labels[i]
         k = k+1
         xlua.progress(k, self.train_data.data.size)
      end
   end

   k=1
   for batch,labels,n in self.test_data:iterator() do
      self.batch = self.batch or batch:transpose(2,3):contiguous():type(self.model:type())

      self.batch:copy(batch:transpose(2, 3):contiguous())

      if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
      self.output = self.new_model:forward(self.batch)
      for i=1, labels:size()[1] do
         if k > self.test_data.data.size then
            break
         end
         test_inputs[k] = self.output[i]
         test_labels[k] = labels[i]
         k = k+1
         xlua.progress(k, self.test_data.data.size)
      end
   end

   torch.save('train.vec', train_inputs)
   torch.save('train.lab', train_labels)

   torch.save('test.vec', test_inputs)
   torch.save('test.lab', test_labels)

end
