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



error(SamplePoints,ws,bs) = 
let 
err = 0
for s in SamplePoints
	err += (s[1]-GenericNN(s[2],ws,bs))^2
end
return err
end



arbitraryNN(inputNum,trainingData) = 
let

W = randn(inputNum*(inputNum+1))
b = randn(inputNum+1)


e = error(trainingData,W,b)
println(e)
println(" ")

n = 0

while n < 1000
	gs = gradient(() -> error(trainingData,W,b), params(W,b))

	newws = gs[W]
	newbs = gs[b]
	W .-= 0.1 .* newws
	b .-= 0.1 .* newbs
	
	e = error(trainingData,W,b)
	
	n+=1
end

return (x->GenericNN(x,W,b))

end


NN = arbitraryNN(2, [(0, [0,0]), (0,[1,1]), (1,[1,0]), (1,[0,1])])

println(NN([1,1]))

println(NN([1,0]))
println(NN([0,1]))
println(NN([0,0]))



