import evaluate as EV
comps = [(m, "main") for m in ["LandmarkLR", "CSFusion", "GRUD", "LTMD", "LTMD-Ens", "CTFilter"]]
EV.bootstrap(("ULTRA", "main"), comps, split="test", out="runs/boot_test.json")
EV.bootstrap(("ULTRA", "main"), comps, split="extE", out="runs/boot_extE.json")
EV.bootstrap(("ULTRA", "main"), comps, split="extD", out="runs/boot_extD.json")
abl = [("ULTRA", t) for t in ["no_align", "no_factor", "no_aging", "no_velocity", "homosc", "no_integrate", "no_innov", "no_axis"]]
EV.bootstrap(("ULTRA", "main"), abl, split="test", out="runs/boot_abl.json")
print("boot done")
