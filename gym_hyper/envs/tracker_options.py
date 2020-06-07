class Opts:
    def __init__(self,  # scale_step, scale_penalty, scale_lr, window_influence,
                 hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h,
                 final_score_sz, filename, image, templates_z, templates_z_, z_sz, scores,
                 re_sz, scores_low):
        # self.scale_step = scale_step
        # self.scale_penalty = scale_penalty
        # self.scale_lr = scale_lr
        # self.window_influence = window_influence
        self.hp = hp
        self.run = run
        self.design = design
        self.frame_name_list = frame_name_list
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.target_w = target_w
        self.target_h = target_h
        self.final_score_sz = final_score_sz
        self.filename = filename
        self.image = image
        self.templates_z = templates_z
        self.templates_z_ = templates_z_
        self.z_sz = z_sz
        self.scores = scores
        self.re_sz = re_sz
        self.scores_low = scores_low
        # self.start_frame = start_frame
