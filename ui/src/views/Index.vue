<template>
  <div>
    <el-row>
      <div class="index-div-title">{{ title }}</div>
    </el-row>
    <el-row>
      <el-col :span="2">
        <el-menu default-active="0" @select="changeMode">
          <el-menu-item
            v-for="(item, index) in naviTitle"
            :key="item"
            :index="String(index)"
          >
            <span slot="title">{{ item }}</span>
          </el-menu-item>
        </el-menu>
      </el-col>
      <el-col :span="20">
        <div class="index-video-box">
          <img id="index-video-frame-img" :src="playVideoBg" />
          <img
            class="index-video-switch"
            :src="switchVideoIcon"
            @click="videoSwitch"
          />
        </div>
      </el-col>
      <el-col :span="2">
        <el-tag
          class="index-body-class-tag"
          v-for="item in bodyClassList"
          :key="'index-body-class-tag-' + item.label"
          :type="item.type"
          :effect="item.effect"
        >
          {{ item.cname }}
        </el-tag>
      </el-col>
    </el-row>
    <el-row>
      <div class="index-statistic-div">
        <el-collapse v-model="activeNames">
          <el-collapse-item title="统计" name="statistic">
            <el-table :data="labelStatistic">
              <el-table-column
                v-for="(item, index) in labelTableTitleList"
                :key="'index-statistic-' + index"
                :prop="item.name"
                :label="item.cname"
              />
            </el-table>
          </el-collapse-item>
        </el-collapse>
      </div>
    </el-row>
    <el-row>
      <div class="index-history-div">
        <el-collapse v-model="activeNames">
          <el-collapse-item title="历史记录" name="history">
            <el-table :data="history" stripe height="400">
              <el-table-column
                v-for="(item, index) in historyTableTitleList"
                :key="'history-table-title-' + index"
                :prop="item.name"
                :label="item.cname"
                width="100"
              />
            </el-table>
          </el-collapse-item>
        </el-collapse>
      </div>
    </el-row>
  </div>
</template>

<script>
import Moment from "moment";
import jquery from "jquery";

export default {
  data() {
    return {
      title: "驾驶员姿态检测系统",
      naviTitle: ["源视频", "关节点"],
      bodyClassList: [
        {
          label: "safe_driving",
          type: "primary",
          effect: "plain",
          cname: "安全驾驶",
        },
        { label: "texting", type: "success", effect: "plain", cname: "打字" },
        {
          label: "talking_on_phone",
          type: "info",
          effect: "plain",
          cname: "打电话",
        },
        { label: "drinking", type: "warning", effect: "plain", cname: "喝水" },
        {
          label: "reaching_behind",
          type: "danger",
          effect: "plain",
          cname: "从后方拿",
        },
        {
          label: "reaching_nearby",
          type: "primary",
          effect: "plain",
          cname: "从身边拿",
        },
        {
          label: "hair_and_makeup",
          type: "success",
          effect: "plain",
          cname: "梳头",
        },
        { label: "tired", type: "info", effect: "plain", cname: "打哈欠" },
        {
          label: "operating_radio",
          type: "warning",
          effect: "plain",
          cname: "调收音机",
        },
      ],
      labelDict: {
        initializing: "初始化",
        safe_driving: "安全驾驶",
        texting: "打字",
        talking_on_phone: "打电话",
        drinking: "喝水",
        reaching_behind: "从后方拿",
        reaching_nearby: "从身边拿",
        hair_and_makeup: "梳头",
        tired: "打哈欠",
        operating_radio: "调收音机",
      },
      labelStatistic: [
        {
          safe_driving: 0,
          texting: 0,
          talking_on_phone: 0,
          drinking: 0,
          reaching_behind: 0,
          reaching_nearby: 0,
          hair_and_makeup: 0,
          tired: 0,
          operating_radio: 0,
        },
      ],
      labelTableTitleList: [
        { name: "safe_driving", cname: "安全驾驶" },
        { name: "texting", cname: "打字" },
        { name: "talking_on_phone", cname: "打电话" },
        { name: "drinking", cname: "喝水" },
        { name: "reaching_behind", cname: "从后方拿" },
        { name: "reaching_nearby", cname: "从身边拿" },
        { name: "hair_and_makeup", cname: "梳头" },
        { name: "tired", cname: "打哈欠" },
        { name: "operating_radio", cname: "调收音机" },
      ],
      bodyClassIndexDict: {
        safe_driving: 0,
        texting: 1,
        talking_on_phone: 2,
        drinking: 3,
        reaching_behind: 4,
        reaching_nearby: 5,
        hair_and_makeup: 6,
        tired: 7,
        operating_radio: 8,
      },
      bodyPointsList: [
        "left-shoulder",
        "left-elbow",
        "left-wrist",
        "right-shoulder",
        "right-elbow",
        "right-wrist",
        "mouse",
        "right-ear",
        "wheel",
      ],
      activeNames: [],
      isVideoPlaying: false,
      shouldShowPoints: false,
      history: [],
      historyTableTitleList: [
        { name: "time", cname: "时间" },
        { name: "label", cname: "动作" },
        { name: "left-shoulder", cname: "左肩膀" },
        { name: "left-elbow", cname: "左肘" },
        { name: "left-wrist", cname: "左手腕" },
        { name: "right-shoulder", cname: "右肩膀" },
        { name: "right-elbow", cname: "右肘" },
        { name: "right-wrist", cname: "右手腕" },
        { name: "mouse", cname: "嘴" },
        { name: "right-ear", cname: "右耳" },
        { name: "wheel", cname: "方向盘" },
      ],
      playVideoBg: "/static/bg-640x480.png",
      playVideoIcon: "/static/video_play_icon.png",
      pauseVideoIcon: "/static/video_pause_icon.png",
      switchVideoIcon: "/static/video_play_icon.png",
    };
  },
  mounted() {
    let videoBox = jquery(".index-video-box");
    let switchIcon = jquery(".index-video-switch");
    videoBox.mouseover((e) => {
      if (this.isVideoPlaying) {
        switchIcon.show();
      }
    });
    videoBox.mouseout((e) => {
      if (this.isVideoPlaying) {
        switchIcon.hide();
      }
    });
  },
  methods: {
    async videoSwitch() {
      if (this.isVideoPlaying) {
        this.switchVideoIcon = this.playVideoIcon;
        this.isVideoPlaying = false;
      } else {
        this.switchVideoIcon = this.pauseVideoIcon;
        this.isVideoPlaying = true;
        while (this.isVideoPlaying) {
          await this.getVideoFrame();
        }
      }
      // let timer = setInterval(async () => {
      //   if (this.isVideoPlaying) {
      //     await this.getVideoFrame()
      //   }
      //   else {
      //     clearInterval(timer);
      //   }
      // }, 50);
    },
    changeMode(key, keyPath) {
      if (key == "0") {
        this.shouldShowPoints = false;
      } else {
        this.shouldShowPoints = true;
      }
    },
    time() {
      let date = new Moment(Date.parse(Date()));
      return date.format("HH:mm:ss");
    },
    updateTime() {
      setInterval(() => (this.now = this.time()), 1000);
    },
    genHistory(label, points) {
      let historyItem = {
        time: this.time(),
        label: this.labelDict[label],
      };
      for (let i = 0; i < this.bodyPointsList.length; i++) {
        if (points[i][0] != -1 && points[i][1] != -1) {
          historyItem[this.bodyPointsList[i]] =
            "x: " + points[i][0] + " y: " + points[i][1];
        } else {
          historyItem[this.bodyPointsList[i]] = "undetected";
        }
      }
      this.history.push(historyItem);
    },
    updateStatistic(label) {
      this.labelStatistic[0][label] += 1;
    },
    async getVideoFrame() {
      await this.axios
        .get("/get_video_frame", {
          params: {
            should_show_points: this.shouldShowPoints,
          },
        })
        .then(async (res) => {
          let data = res.data;
          if (data.success) {
            let imgBlob = await (await fetch(data.imgBytes)).blob();
            let bufferImage = new Image();
            bufferImage.src = URL.createObjectURL(imgBlob);
            document.getElementById("index-video-frame-img").src =
              bufferImage.src;
            if (data.label != "initializing") {
              for (let i = 0; i < this.bodyClassList.length; i++) {
                if (i != this.bodyClassIndexDict[data.label]) {
                  this.bodyClassList[i].effect = "plain";
                } else {
                  this.bodyClassList[i].effect = "dark";
                }
              }
            }
            if (data.isNewPoints) {
              this.genHistory(data.label, data.points);
              if (data.label != "initializing") {
                this.updateStatistic(data.label);
              }
            }
            if (data.isEnd) {
              this.isVideoPlaying = false;
            }
          }
        });
    },
  },
};
</script>

<style>
.index-div-title {
  margin-bottom: 20px;
  font: 20px Microsoft YaHei;
}
.index-video-box {
  width: 832px;
  height: 624px;
  margin: 0 auto 0 auto;
}
#index-video-frame-img {
  width: 100%;
  height: 100%;
}
.index-video-switch {
  position: absolute;
  margin: auto;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  cursor: pointer;
}
.index-body-class-tag {
  width: 90px;
  margin: 10px 0 10px 0;
}
.index-statistic-div {
  margin-top: 20px;
}
.index-history-div {
  margin-top: 20px;
}
</style>
