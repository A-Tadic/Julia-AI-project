#First what I'll do is program this up using only Zygote, then I'll do it all with Flux

using Zygote
using Flux



sigmoid(x) = 1/(1+exp(-x))

neuron(xs,ws,wstart,b) =
let
i = 1
sum = b

while i <= length(xs)
	sum += (xs[i]*ws[wstart+i])
	i +=1
end
return sigmoid(sum)
end	



GenericNN(xs,ws,bs) =
let 


i = 1
output = bs[length(bs)]


while i <= length(xs)
	n = neuron(xs,ws,(i-1)*length(xs),bs[i])
	output += n*ws[length(xs)^2+i]
	i+= 1
end


	
return sigmoid(output)
end



GenericNN_DEEP_LEARNING(xs,ws,bs) =
let 


i = length(xs)+1

j = 1

output = bs[length(bs)]




while i <= 2*length(xs)

	ws2 = vcat(ws[1:length(xs)^2], ws[i:(i+length(xs))])
	bs2 = vcat(bs[1:length(xs)], bs[i])
	

	n = GenericNN(xs,ws2,bs2)
	output += n*ws[2*length(xs)^2+j]  
	i+= 1
	j += 1
end


	
return sigmoid(output)
end


error(SamplePoints,ws,bs) = 
let 
err = 0
for s in SamplePoints
	err += (s[1]-GenericNN(s[2],ws,bs))^2
end
return err
end



error_DEEP_LEARNING(SamplePoints,ws,bs) = 
let 
err = 0
for s in SamplePoints
	err += (s[1]-GenericNN_DEEP_LEARNING(s[2],ws,bs))^2
end
return err
end


#Taken from https://discourse.julialang.org/t/sampling-without-replacement/1073/5
sample_wo_repl(A,n) =
let
    sample = []
    for i in 1:n
       push!(sample, splice!(A, rand(eachindex(A))))
    end
    return sample
end


arbitraryNN_SGD(inputNum,trainingData) = 
let



W = randn(inputNum*(inputNum+1))
b = randn(inputNum+1)



n = 0

while n < 1000

	copy = deepcopy(trainingData)

	sample = sample_wo_repl(copy,4)\
	
	
	for i in sample
	
		gs = gradient(() -> error([i],W,b), params(W,b))

		newws = gs[W]
		newbs = gs[b]
		W .-= 2 .* newws
		b .-= 2 .* newbs
	end

	
	
	n+=1
end

return (x->GenericNN(x,W,b))

end


arbitraryNN_minibatch(inputNum,trainingData) = 
let



W = randn(inputNum*(inputNum+1))
b = randn(inputNum+1)



n = 0

while n < 1000

	copy = deepcopy(trainingData)

	
	batchsize = 3
	
	while length(copy) > 0
	
		
	
		if length(copy) >= batchsize
			sample = sample_wo_repl(copy,batchsize)
		else
			sample = copy
			copy = []
		end
		
		#println(sample)
		
		
		gs = gradient(() -> error(sample,W,b), params(W,b))

		newws = gs[W]
		newbs = gs[b]
		W .-= 2 .* newws
		b .-= 2 .* newbs
	end

	#println()
	
	n+=1
end

return (x->GenericNN(x,W,b))

end


arbitraryNN_momentum(inputNum,trainingData) = 
let



W = randn(inputNum*(inputNum+1))
b = randn(inputNum+1)

prevgradW = zeros(length(W))
prevgradb = zeros(length(b))

n = 0

while n < 175#3 and 200 works


	gs = gradient(() -> error(trainingData,W,b), params(W,b))#try making a parametrized gradient function. This should be doable outside of the while loop

	newws = gs[W]
	newbs = gs[b]
	W .-= 2 .* newws + 0.9(prevgradW)
	b .-= 2 .* newbs + 0.9(prevgradb)
	
	prevgradW = newws
	prevgradb = newbs
	
	
	n+=1
end

return (x->GenericNN(x,W,b))

end



arbitraryNN_SGD_momentum(inputNum,trainingData) = 
let



W = randn(inputNum*(inputNum+1))
b = randn(inputNum+1)

prevgradW = zeros(length(W))
prevgradb = zeros(length(b))


n = 0

while n < 200

	copy = deepcopy(trainingData)

	sample = sample_wo_repl(copy,4)#momentum really starts to work with batch size 3
	#Taking a batch size of 2 works pretty well
	
	
	for i in sample
		gs = gradient(() -> error(sample,W,b), params(W,b))

		newws = gs[W]
		newbs = gs[b]
		W .-= 2 .* newws + 0.9(prevgradW)
		b .-= 2 .* newbs + 0.9(prevgradb)
		
		prevgradW = newws
		prevgradb = newbs
		
	end

	
	
	n+=1
end

return (x->GenericNN(x,W,b))

end

#This seems to work consistently for AND with 200 iterations
#I tried it once w/ 150 iterations (AND). it worked once and failed once.
#Seems to work slightly better then SGD w/ momentum whenever it works
#I think the weirdness with XOR is on account of some strange property of the function rather than an 
#error in the code.
arbitraryNN_SGD_NAG(inputNum,trainingData) = 
let



W = randn(inputNum*(inputNum+1))
b = randn(inputNum+1)

vw = zeros(length(W))
vb = zeros(length(b))


n = 0

learningRate = 2
momentum = 0.9

while n < 150

	copy = deepcopy(trainingData)

	sample = sample_wo_repl(copy,4)
	
	
	for i in sample
	
		w1 = W - momentum*vw
		b1 = b - momentum*vb
	
		gs = gradient(() -> error([i],w1,b1), params(w1,b1))

		newVw = momentum*vw + learningRate*gs[w1]
		newVb = momentum*vb + learningRate*gs[b1]
		
		W = W - newVw
		b = b - newVb
		
		vw = newVw
		vb = newVb
	end

	
	
	n+=1
end

return (x->GenericNN(x,W,b))

end


arbitraryNN_minibatch_NAG(inputNum,trainingData) = 
let


W = randn(inputNum*(inputNum+1))
b = randn(inputNum+1)

vw = zeros(length(W))
vb = zeros(length(b))


n = 0

learningRate = 2
momentum = 0.9
batchsize = 4
	
	
while n < 100

	copy = deepcopy(trainingData)

	sample = sample_wo_repl(copy,4)
	
	

	
	while length(copy) > 0
	
		
	
		if length(copy) >= batchsize
			sample = sample_wo_repl(copy,batchsize)
		else
			sample = copy
			copy = []
		end
		
	
		w1 = W - momentum*vw
		b1 = b - momentum*vb
	
		gs = gradient(() -> error(sample,w1,b1), params(w1,b1))

		newVw = momentum*vw + learningRate*gs[w1]
		newVb = momentum*vb + learningRate*gs[b1]
		
		W = W - newVw
		b = b - newVb
		
		vw = newVw
		vb = newVb
	
	end
	
	
	n+=1
end

return (x->GenericNN(x,W,b))

end


#for some reason, my NAG stuff ultimately goes to
#$0.5004066111171824
#0.9876557658668574
#0.49986085209694103
#0.00919037038556629
#w/ XOR while it works the rest of the time...
arbitraryNN_NAG(inputNum,trainingData) = 
let



W = randn(inputNum*(inputNum+1))
b = randn(inputNum+1)

println(W)
println()
println(b)
println()


	
gs = gradient(() -> error(trainingData,W,b), params(W,b))

vw = 2 .*gs[W]
vb = 2 .*gs[b]
W .-=  vw
b .-=  vb
	

#println(vw)
#println(vb)	
#println()

n = 1

while n <200


	
	w1 =  W-0.9*vw
	b1 = b - 0.9*vb
	
	gs = gradient(() -> error(trainingData,w1,b1), params(w1,b1))

	vw = 0.9*vw+2*gs[w1]
	vb = 0.9*vb+2*gs[b1]
	
	#println(vw)
#println(vb)	
#println()
	
	W = W - vw
	b = b - vb
	
	
	
	
	
	n+=1
end

return (x->GenericNN(x,W,b))

end




arbitraryNN_basicAdagrad(inputNum,trainingData) = 
let



W = randn(inputNum*(inputNum+1))
b = randn(inputNum+1)


sumOfGradSquaresW = zeros(length(W))
sumOfGradSquaresb = zeros(length(b))


n = 0

while n < 1000

	copy = deepcopy(trainingData)

	sample = sample_wo_repl(copy,4)#momentum really starts to work with batch size 3
	#Taking a batch size of 2 works pretty well
	
	gs = gradient(() -> error(sample,W,b), params(W,b))


	wc= deepcopy(gs[W])
	bc= deepcopy(gs[b])
	
	for i in 1:length(W)
		W[i] = W[i] - (2/sqrt(sumOfGradSquaresW[i]+(0.01)))*gs[W][i]
	end
	
	for i in 1:length(b)
		b[i] = b[i] - (2/sqrt(sumOfGradSquaresb[i]+(0.01)))*gs[b][i]
	end
	

	
	for i in 1:length(W)
		wc[i] = wc[i]^2
	end
	
	for i in 1:length(b)
		bc[i] = bc[i]^2
	end
	
	sumOfGradSquaresW = sumOfGradSquaresW + wc
	sumOfGradSquaresb = sumOfGradSquaresb + bc

	
	
	n+=1
end

return (x->GenericNN(x,W,b))

end

arbitraryNN_SGD_adagrad(inputNum,trainingData) = 
let



W = randn(inputNum*(inputNum+1))
b = randn(inputNum+1)



n = 0

while n < 1000

	copy = deepcopy(trainingData)

	sample = sample_wo_repl(copy,4)#momentum really starts to work with batch size 3
	#Taking a batch size of 2 works pretty well
	
	
	for i in sample
		gs = gradient(() -> error(sample,W,b), params(W,b))

		newws = gs[W]
		newbs = gs[b]
		W .-= 2 .* newws
		b .-= 2 .* newbs
	end

	
	
	n+=1
end

return (x->GenericNN(x,W,b))

end


arbitraryNN_DEEP_LEARNING(inputNum,trainingData) = 
let

W = randn(2*inputNum*inputNum+inputNum)
b = randn(2*inputNum+1)


e = error_DEEP_LEARNING(trainingData,W,b)
println(e)
println(" ")

n = 0

while n < 500
	gs = gradient(() -> error_DEEP_LEARNING(trainingData,W,b), params(W,b))

	newws = gs[W]
	newbs = gs[b]
	W .-= 2 .* newws
	b .-= 2 .* newbs
	
	e = error_DEEP_LEARNING(trainingData,W,b)
	
	n+=1
end

return (x->GenericNN_DEEP_LEARNING(x,W,b))

end




NN = arbitraryNN_SGD_momentum(2, [(0,[0,0]),(0,[1,0]),(0,[0,1]),(1,[1,1])] )

println(NN([1,1]))

println(NN([1,0]))
println(NN([0,1]))
println(NN([0,0]))




