from pxr import Usd


ASSET_DIR = "/home/crtie/Downloads/factory_hole_8mm.usd"




# stage = Usd.Stage.Open("/home/crtie/crtie/Manual2Skill2/chair_real/frame.usd")
stage = Usd.Stage.Open(ASSET_DIR)
for prim in stage.Traverse():
    print(f"Prim: {prim.GetPath()}  Type: {prim.GetTypeName()}")
    # 打印属性
    for attr in prim.GetAttributes():
        print(f"  Attribute: {attr.GetName()}  Type: {attr.GetTypeName()}")
    # 打印已应用的API
    for api in prim.GetAppliedSchemas():
        print(f"  Applied API: {api}")
