import docsaidkit as D
from docaligner import DocAligner, ModelType

IPADDR = '192.168.0.179'  # Change this to your IP camera address

service1 = DocAligner(model_type=ModelType.heatmap, model_cfg='fastvit_sa24')


def _run(img):
    result1 = service1(img, do_center_crop=False)
    if result1.has_doc_polygon:
        img = D.draw_polygon(img, result1.doc_polygon,
                             color=(0, 0, 255), thickness=2)
    return img


demo = D.WebDemo(IPADDR, pipelines=[_run])
demo.run()
