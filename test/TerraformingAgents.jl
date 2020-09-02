# import TerraformingAgents
using TerraformingAgents

@testset "Check argument lengths" begin 

    args = (pos = [(1, 2), (3.3, 2)], 
            vel = [(1.1, 2.2), (0.1, 2)], 
            planetcompositions = [[[1,2,3],[4,5,6]],
                                  [[5,6,7],[8,9,10]]])
    
    @test TerraformingAgents.haveidenticallengths(args)

    @test_throws

end

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