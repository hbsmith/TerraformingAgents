# import TerraformingAgents
using TerraformingAgents

@testset "Check provided args includes non-nothing value" begin
    
    argdict = Dict(
        :pos => [(1, 2), (3.3, 2)], 
        :vel => [(1.1, 2.2), (0.1, 2)], 
        :planetcompositions => [[[1,2,3],[4,5,6]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == argdict

    ## planetcompositions with different lengths
    argdict = Dict(
        :pos => [(1, 2), (3.3, 2)], 
        :vel => [(1.1, 2.2), (0.1, 2)], 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == argdict

    ## various nothing args
    argdict = Dict(
        :pos => nothing, 
        :vel => nothing, 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == Dict(:planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    argdict = Dict(
        :pos => [(1, 2), (3.3, 2)], 
        :vel => nothing, 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test TerraformingAgents.providedargs(argdict) == Dict( :pos => [(1, 2), (3.3, 2)], :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])

    @test_throws ArgumentError TerraformingAgents.providedargs(Dict(
        :pos => nothing, 
        :vel => nothing, 
        :planetcompositions => nothing))

end

@testset "Check if args have identical lengths" begin
    
    @test TerraformingAgents.haveidenticallengths(Dict(
        :pos => [(1, 2), (3,4)], 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]]))

    ## Length mismatches                                                                   
    @test TerraformingAgents.haveidenticallengths(Dict(
        :pos => [(1, 2)], 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])) == false

    @test TerraformingAgents.haveidenticallengths(Dict(
        :pos => [(1, 2),(3,4.1)], 
        :vel => [(1, 2)], 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])) == false
        
    @test_throws MethodError TerraformingAgents.haveidenticallengths(Dict(
        :pos => [(1, 2),(3,4.1)], 
        :vel => nothing, 
        :planetcompositions => [[[1,2,3]], [[5,6,7],[8,9,10]]])) 

end

# @testset "Check argument lengths" begin 

#     @test TerraformingAgents.haveidenticallengths((
#         pos = [(1, 2), (3.3, 2)], 
#         vel = [(1.1, 2.2), (0.1, 2)], 
#         planetcompositions = [[[1,2,3],[4,5,6]], [[5,6,7],[8,9,10]]]))
    
#     ## planetcompositions with different lengths
#     @test TerraformingAgents.haveidenticallengths((
#         pos = [(1, 2), (3.3, 2)], 
#         vel = [(1.1, 2.2), (0.1, 2)], 
#         planetcompositions = [[[1,2,3]], [[5,6,7],[8,9,10]]]))

#     ## various nothing args
#     @test TerraformingAgents.haveidenticallengths((
#         pos = nothing, 
#         vel = nothing, 
#         planetcompositions = [[[1,2,3]], [[5,6,7],[8,9,10]]]))

#     @test TerraformingAgents.haveidenticallengths((
#         pos = nothing, 
#         vel = [(1.1, 2.2), (0.1, 2)], 
#         planetcompositions = nothing))

#     @test TerraformingAgents.haveidenticallengths((
#         pos = [(1, 2), (3.3, 2)], 
#         vel = nothing, 
#         planetcompositions = [[[1,2,3]], [[5,6,7],[8,9,10]]]))
    
#     @test_throws ArgumentError TerraformingAgents.haveidenticallengths((
#         pos = nothing, 
#         vel = nothing, 
#         planetcompositions = nothing))
     
#     ## Length mismatches                                                                   
#     @test_throws ArgumentError TerraformingAgents.haveidenticallengths((
#         pos = [(1, 2)], 
#         vel = nothing, 
#         planetcompositions = [[[1,2,3]], [[5,6,7],[8,9,10]]]))

#     @test_throws ArgumentError TerraformingAgents.haveidenticallengths((
#         pos = [(1, 2),(3,4.1)], 
#         vel = [(1, 2)], 
#         planetcompositions = [[[1,2,3]], [[5,6,7],[8,9,10]]]))                                                                        
# end

# @testset "Initialization" begin

#     args = Dict(:nplanetarysystems => 10)

#     modelparams = Dict(:RNG => MersenneTwister(1236),
#                     :psneighbor_radius => .45,
#                     :dt => 0.1)

#     model = galaxy_model(
#         args[:nplanetarysystems]
#         ;modelparams...)

#     ## give pos, use default vel, comp
#     ## give vel, use defualt pos, comp
#     ## give pos and vel, use default comp 
#     ## give pos and comp, use default vel 
#     ## give 


# end

# @testset "sort_biosystem_compounds" begin

#     compounds = readcompounds("data/compound")

#     @testset "sortkey: exact_mass" begin

#         biosystem_compounds = readids(["C00068","C00069","C00379","C00380","C00381"])
#         sortkey = :exact_mass
#         zero_mass_behavior = "end"

#         expected_exact_mass = [("C00380",111.0433),
#                                ("C00379",152.0685),
#                                ("C00068",425.045),
#                                ("C00381",0.0),
#                                ("C00069",0.0)]

#         Random.seed!(1234);
#         @test expected_exact_mass == BioXP.sort_biosystem_compounds(compounds,
#                                                     biosystem_compounds,
#                                                     sortkey,
#                                                     zero_mass_behavior)
        
#         for i in 1:1000
#             @test expected_exact_mass[1:3] == BioXP.sort_biosystem_compounds(compounds,
#                                                     biosystem_compounds,
#                                                     sortkey,
#                                                     zero_mass_behavior)[1:3]

#         end
        
    # end