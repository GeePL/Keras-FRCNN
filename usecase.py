%run tool.py
path = './temp1/'
#导入模型
detector=frcnn(path+'acc_model.hdf5',path+'config.pickle')
#调用detectSaveByXml函数，功能为读入图片和对应的xml文件，返回模型预测的结果与实际结果（即每张图片的测试类别，实际类别），以及总体的准确率
#voc_path即数据集路径，save_all为保存所有预测结果图片，save_failed为保存预测失败的图片，save_path为保存路径，img_set为选择数据集中的什么集合作为测试，zero_path为正常图片集合，bbox_threshold为预测结果判病疵的阈值
pre_vol,real_vol,acc_vol = detector.detectSaveByXml(voc_path='/cy_whole/voc/',save_all=True,save_failed=True,
save_path=path+'voc/', img_set = 'test', zero_path = '/cy_whole/zeros/zero_test/', bbox_threshold=0.6)
print acc_vol
#调用getResultFromRealPre函数，输入预测结果和实际结果，返回各类别的召回率和准确率，并生成对应的一张图
ignoreClass=['65','66','68','69','70','78','79','80','65','74','76']
tp=detector.getResultFromRealPre(pre=pre_vol, real=real_vol,ignoreClass=ignoreClass)