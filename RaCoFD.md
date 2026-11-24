rtdetrv4代码记录：
engine\rtv4\hybrid_encoder.py
    通过线性方式学生模型映射到dino特征空间[b 256 20 20] -> [b 768 20 20]
        if self.training and self.feature_projector is not None and enc_ind == self.encoder_idx_for_distillation:
            # _logger.info(f"[HybridEncoder] Feature size: {h}x{w}")
            distill_student_output = self.feature_projector(proj_feats[enc_ind].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)







新论文架构设计：
    engine\solver\det_engine.py
        if teacher_model is not None:
            with torch.no_grad(): # 走dinov3的教师模型,sample就是rtdetr的图像输入
                teacher_encoder_output_for_distillation = teacher_model(samples).detach()
        再DINO教师模型下添加teacher_detect_model的大尺度检测模型，要求和DINO格式一样，如：
        if teacher_detect_model is not None:
            with torch.no_grad(): # 走dinov3的教师模型,sample就是rtdetr的图像输入
                teacher_detect_encoder_output_for_distillation = teacher_detect_model(samples).detach()
    configs文件夹下构建1个racofd的文件夹，类似rtv4，包含coco_s尺度配置


                