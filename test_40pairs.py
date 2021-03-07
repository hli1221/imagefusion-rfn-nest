# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: hui_li_jnu@163.com
# @File : test_40pairs.py
# @Time : 2020/8/14 17:11

# test phase
import os
import torch
from torch.autograd import Variable
from net import NestFuse_light2_nodense, Fusion_network, Fusion_strategy
import utils
from args_fusion import args
import numpy as np


def load_model(path_auto, path_fusion, fs_type, flag_img):
	if flag_img is True:
		nc = 3
	else:
		nc =1
	input_nc = nc
	output_nc = nc
	nb_filter = [64, 112, 160, 208, 256]

	nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision=False)
	nest_model.load_state_dict(torch.load(path_auto))

	fusion_model = Fusion_network(nb_filter, fs_type)
	fusion_model.load_state_dict(torch.load(path_fusion))

	fusion_strategy = Fusion_strategy(fs_type)

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	para = sum([np.prod(list(p.size())) for p in fusion_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(fusion_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	fusion_model.eval()
	nest_model.cuda()
	fusion_model.cuda()

	return nest_model, fusion_model, fusion_strategy


def run_demo(nest_model, fusion_model, fusion_strategy, infrared_path, visible_path, output_path_root, name_ir, fs_type, use_strategy, flag_img, alpha):
	img_ir, h, w, c = utils.get_test_image(infrared_path, flag=flag_img)  # True for rgb
	img_vi, h, w, c = utils.get_test_image(visible_path, flag=flag_img)

	# dim = img_ir.shape
	if c is 1:
		if args.cuda:
			img_ir = img_ir.cuda()
			img_vi = img_vi.cuda()
		img_ir = Variable(img_ir, requires_grad=False)
		img_vi = Variable(img_vi, requires_grad=False)
		# encoder
		en_r = nest_model.encoder(img_ir)
		en_v = nest_model.encoder(img_vi)
		# fusion net
		if use_strategy:
			f = fusion_strategy(en_r, en_v)
		else:
			f = fusion_model(en_r, en_v)
		# decoder
		img_fusion_list = nest_model.decoder_eval(f)
	else:
		# fusion each block
		img_fusion_blocks = []
		for i in range(c):
			# encoder
			img_vi_temp = img_vi[i]
			img_ir_temp = img_ir[i]
			if args.cuda:
				img_vi_temp = img_vi_temp.cuda()
				img_ir_temp = img_ir_temp.cuda()
			img_vi_temp = Variable(img_vi_temp, requires_grad=False)
			img_ir_temp = Variable(img_ir_temp, requires_grad=False)

			en_r = nest_model.encoder(img_ir_temp)
			en_v = nest_model.encoder(img_vi_temp)
			# fusion net
			if use_strategy:
				f = fusion_strategy(en_r, en_v)
			else:
				f = fusion_model(en_r, en_v)
			# decoder
			img_fusion_temp = nest_model.decoder_eval(f)
			img_fusion_blocks.append(img_fusion_temp)
		img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

	# ########################### multi-outputs ##############################################
	output_count = 0
	for img_fusion in img_fusion_list:
		file_name = 'fused_' + str(alpha) + '_' + name_ir
		output_path = output_path_root + file_name
		output_count += 1
		# save images
		utils.save_image_test(img_fusion, output_path)
		print(output_path)


def main():
	# False - gray
	flag_img = False
	# ################# gray scale ########################################
	test_path = "images/40_pairs_tno_vot/ir/"
	path_auto = args.resume_nestfuse
	output_path_root = "./outputs/alpha_1e4_40/"
	if os.path.exists(output_path_root) is False:
		os.mkdir(output_path_root)

	fs_type = 'res'  # res (RFN), add, avg, max, spa, nuclear
	use_strategy = False  # True - static strategy; False - RFN

	path_fusion_root = args.fusion_model

	with torch.no_grad():
		alpha_list = [700]
		w_all_list = [[6.0, 3.0]]

		for alpha in alpha_list:
			for w_all in w_all_list:
				w, w2 = w_all

				temp = 'rfnnest_' + str(alpha) + '_wir_' + str(w) + '_wvi_' + str(w2)
				output_path_list = 'fused_' + temp + '_40'
				output_path1 = output_path_root + output_path_list + '/'
				if os.path.exists(output_path1) is False:
					os.mkdir(output_path1)
				output_path = output_path1
				# load network
				path_fusion = path_fusion_root + str(w) + '/' + 'Final_epoch_2_alpha_' + str(alpha) + '_wir_' + str(
					w) + '_wvi_' + str(w2) + '_ssim_vi.model'
				model, fusion_model, fusion_strategy = load_model(path_auto, path_fusion, fs_type, flag_img)
				imgs_paths_ir, names = utils.list_images(test_path)
				num = len(imgs_paths_ir)
				for i in range(num):
					name_ir = names[i]
					infrared_path = imgs_paths_ir[i]
					visible_path = infrared_path.replace('ir/', 'vis/')
					if visible_path.__contains__('IR'):
						visible_path = visible_path.replace('IR', 'VIS')
					else:
						visible_path = visible_path.replace('i.', 'v.')
					run_demo(model, fusion_model, fusion_strategy, infrared_path, visible_path, output_path, name_ir, fs_type, use_strategy, flag_img, temp)
				print('Done......')


if __name__ == '__main__':
	main()
