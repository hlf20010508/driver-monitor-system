webpackJsonp([1],{"3eKh":function(e,t){},FhlZ:function(e,t){},NHnr:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var s=n("7+uW"),a={render:function(){var e=this.$createElement,t=this._self._c||e;return t("div",{attrs:{id:"app"}},[t("router-view")],1)},staticRenderFns:[]};var i=n("VU/8")({name:"App"},a,!1,function(e){n("3eKh")},null,null).exports,r=n("/ocq"),l=n("Xxa5"),o=n.n(l),c=n("exGp"),d=n.n(c),m=n("PJh5"),u=n.n(m),h=n("7t+N"),j=n.n(h),p={data:function(){return{title:"驾驶员姿态检测系统",naviTitle:["源视频","关节点"],bodyClassList:[{label:"safe_driving",type:"primary",effect:"plain",cname:"安全驾驶"},{label:"texting",type:"success",effect:"plain",cname:"打字"},{label:"talking_on_phone",type:"info",effect:"plain",cname:"打电话"},{label:"drinking",type:"warning",effect:"plain",cname:"喝水"},{label:"reaching_behind",type:"danger",effect:"plain",cname:"从后方拿"},{label:"reaching_nearby",type:"primary",effect:"plain",cname:"从身边拿"},{label:"hair_and_makeup",type:"success",effect:"plain",cname:"梳头"},{label:"tired",type:"info",effect:"plain",cname:"打哈欠"},{label:"operating_radio",type:"warning",effect:"plain",cname:"调收音机"}],labelDict:{initializing:"初始化",safe_driving:"安全驾驶",texting:"打字",talking_on_phone:"打电话",drinking:"喝水",reaching_behind:"从后方拿",reaching_nearby:"从身边拿",hair_and_makeup:"梳头",tired:"打哈欠",operating_radio:"调收音机"},labelStatistic:[{safe_driving:0,texting:0,talking_on_phone:0,drinking:0,reaching_behind:0,reaching_nearby:0,hair_and_makeup:0,tired:0,operating_radio:0}],labelTableTitleList:[{name:"safe_driving",cname:"安全驾驶"},{name:"texting",cname:"打字"},{name:"talking_on_phone",cname:"打电话"},{name:"drinking",cname:"喝水"},{name:"reaching_behind",cname:"从后方拿"},{name:"reaching_nearby",cname:"从身边拿"},{name:"hair_and_makeup",cname:"梳头"},{name:"tired",cname:"打哈欠"},{name:"operating_radio",cname:"调收音机"}],bodyClassIndexDict:{safe_driving:0,texting:1,talking_on_phone:2,drinking:3,reaching_behind:4,reaching_nearby:5,hair_and_makeup:6,tired:7,operating_radio:8},bodyPointsList:["left-shoulder","left-elbow","left-wrist","right-shoulder","right-elbow","right-wrist","mouse","right-ear","wheel"],activeNames:[],isVideoPlaying:!1,shouldShowPoints:!1,history:[],historyTableTitleList:[{name:"time",cname:"时间"},{name:"label",cname:"动作"},{name:"left-shoulder",cname:"左肩膀"},{name:"left-elbow",cname:"左肘"},{name:"left-wrist",cname:"左手腕"},{name:"right-shoulder",cname:"右肩膀"},{name:"right-elbow",cname:"右肘"},{name:"right-wrist",cname:"右手腕"},{name:"mouse",cname:"嘴"},{name:"right-ear",cname:"右耳"},{name:"wheel",cname:"方向盘"}],playVideoBg:"/static/bg-640x480.png",playVideoIcon:"/static/video_play_icon.png",pauseVideoIcon:"/static/video_pause_icon.png",switchVideoIcon:"/static/video_play_icon.png"}},mounted:function(){var e=this,t=j()(".index-video-box"),n=j()(".index-video-switch");t.mouseover(function(t){e.isVideoPlaying&&n.show()}),t.mouseout(function(t){e.isVideoPlaying&&n.hide()})},methods:{videoSwitch:function(){var e=this;return d()(o.a.mark(function t(){return o.a.wrap(function(t){for(;;)switch(t.prev=t.next){case 0:if(!e.isVideoPlaying){t.next=5;break}e.switchVideoIcon=e.playVideoIcon,e.isVideoPlaying=!1,t.next=12;break;case 5:e.switchVideoIcon=e.pauseVideoIcon,e.isVideoPlaying=!0;case 7:if(!e.isVideoPlaying){t.next=12;break}return t.next=10,e.getVideoFrame();case 10:t.next=7;break;case 12:case"end":return t.stop()}},t,e)}))()},changeMode:function(e,t){this.shouldShowPoints="0"!=e},time:function(){return new u.a(Date.parse(Date())).format("HH:mm:ss")},updateTime:function(){var e=this;setInterval(function(){return e.now=e.time()},1e3)},genHistory:function(e,t){for(var n={time:this.time(),label:this.labelDict[e]},s=0;s<this.bodyPointsList.length;s++)-1!=t[s][0]&&-1!=t[s][1]?n[this.bodyPointsList[s]]="x: "+t[s][0]+" y: "+t[s][1]:n[this.bodyPointsList[s]]="undetected";this.history.push(n)},updateStatistic:function(e){this.labelStatistic[0][e]+=1},getVideoFrame:function(){var e=this;return d()(o.a.mark(function t(){return o.a.wrap(function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,e.axios.get("/get_video_frame",{params:{should_show_points:e.shouldShowPoints}}).then(function(){var t=d()(o.a.mark(function t(n){var s,a,i,r;return o.a.wrap(function(t){for(;;)switch(t.prev=t.next){case 0:if(!(s=n.data).success){t.next=13;break}return t.next=4,fetch(s.imgBytes);case 4:return t.next=6,t.sent.blob();case 6:if(a=t.sent,(i=new Image).src=URL.createObjectURL(a),document.getElementById("index-video-frame-img").src=i.src,"initializing"!=s.label)for(r=0;r<e.bodyClassList.length;r++)r!=e.bodyClassIndexDict[s.label]?e.bodyClassList[r].effect="plain":e.bodyClassList[r].effect="dark";s.isNewPoints&&(e.genHistory(s.label,s.points),"initializing"!=s.label&&e.updateStatistic(s.label)),s.isEnd&&(e.isVideoPlaying=!1);case 13:case"end":return t.stop()}},t,e)}));return function(e){return t.apply(this,arguments)}}());case 2:case"end":return t.stop()}},t,e)}))()}}},f={render:function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",[n("el-row",[n("div",{staticClass:"index-div-title"},[e._v(e._s(e.title))])]),e._v(" "),n("el-row",[n("el-col",{attrs:{span:2}},[n("el-menu",{attrs:{"default-active":"0"},on:{select:e.changeMode}},e._l(e.naviTitle,function(t,s){return n("el-menu-item",{key:t,attrs:{index:String(s)}},[n("span",{attrs:{slot:"title"},slot:"title"},[e._v(e._s(t))])])}),1)],1),e._v(" "),n("el-col",{attrs:{span:20}},[n("div",{staticClass:"index-video-box"},[n("img",{attrs:{id:"index-video-frame-img",src:e.playVideoBg}}),e._v(" "),n("img",{staticClass:"index-video-switch",attrs:{src:e.switchVideoIcon},on:{click:e.videoSwitch}})])]),e._v(" "),n("el-col",{attrs:{span:2}},e._l(e.bodyClassList,function(t){return n("el-tag",{key:"index-body-class-tag-"+t.label,staticClass:"index-body-class-tag",attrs:{type:t.type,effect:t.effect}},[e._v("\n        "+e._s(t.cname)+"\n      ")])}),1)],1),e._v(" "),n("el-row",[n("div",{staticClass:"index-statistic-div"},[n("el-collapse",{model:{value:e.activeNames,callback:function(t){e.activeNames=t},expression:"activeNames"}},[n("el-collapse-item",{attrs:{title:"统计",name:"statistic"}},[n("el-table",{attrs:{data:e.labelStatistic}},e._l(e.labelTableTitleList,function(e,t){return n("el-table-column",{key:"index-statistic-"+t,attrs:{prop:e.name,label:e.cname}})}),1)],1)],1)],1)]),e._v(" "),n("el-row",[n("div",{staticClass:"index-history-div"},[n("el-collapse",{model:{value:e.activeNames,callback:function(t){e.activeNames=t},expression:"activeNames"}},[n("el-collapse-item",{attrs:{title:"历史记录",name:"history"}},[n("el-table",{attrs:{data:e.history,stripe:"",height:"400"}},e._l(e.historyTableTitleList,function(e,t){return n("el-table-column",{key:"history-table-title-"+t,attrs:{prop:e.name,label:e.cname,width:"100"}})}),1)],1)],1)],1)])],1)},staticRenderFns:[]};var g=n("VU/8")(p,f,!1,function(e){n("FhlZ")},null,null).exports;s.default.use(r.a);var b=r.a.prototype.push;r.a.prototype.push=function(e){return b.call(this,e).catch(function(e){return e})};var v=new r.a({routes:[{path:"/",component:g}]}),y=n("zL8q"),k=n.n(y),_=(n("tvR6"),n("mtWM")),w=n("aLYK");s.default.use(k.a),s.default.use(w.a,_.a),s.default.config.productionTip=!1,new s.default({el:"#app",router:v,components:{App:i},template:"<App/>"})},tvR6:function(e,t){},uslO:function(e,t,n){var s={"./af":"3CJN","./af.js":"3CJN","./ar":"3MVc","./ar-dz":"tkWw","./ar-dz.js":"tkWw","./ar-kw":"j8cJ","./ar-kw.js":"j8cJ","./ar-ly":"wPpW","./ar-ly.js":"wPpW","./ar-ma":"dURR","./ar-ma.js":"dURR","./ar-sa":"7OnE","./ar-sa.js":"7OnE","./ar-tn":"BEem","./ar-tn.js":"BEem","./ar.js":"3MVc","./az":"eHwN","./az.js":"eHwN","./be":"3hfc","./be.js":"3hfc","./bg":"lOED","./bg.js":"lOED","./bm":"hng5","./bm.js":"hng5","./bn":"aM0x","./bn-bd":"1C9R","./bn-bd.js":"1C9R","./bn.js":"aM0x","./bo":"w2Hs","./bo.js":"w2Hs","./br":"OSsP","./br.js":"OSsP","./bs":"aqvp","./bs.js":"aqvp","./ca":"wIgY","./ca.js":"wIgY","./cs":"ssxj","./cs.js":"ssxj","./cv":"N3vo","./cv.js":"N3vo","./cy":"ZFGz","./cy.js":"ZFGz","./da":"YBA/","./da.js":"YBA/","./de":"DOkx","./de-at":"8v14","./de-at.js":"8v14","./de-ch":"Frex","./de-ch.js":"Frex","./de.js":"DOkx","./dv":"rIuo","./dv.js":"rIuo","./el":"CFqe","./el.js":"CFqe","./en-au":"Sjoy","./en-au.js":"Sjoy","./en-ca":"Tqun","./en-ca.js":"Tqun","./en-gb":"hPuz","./en-gb.js":"hPuz","./en-ie":"ALEw","./en-ie.js":"ALEw","./en-il":"QZk1","./en-il.js":"QZk1","./en-in":"yJfC","./en-in.js":"yJfC","./en-nz":"dyB6","./en-nz.js":"dyB6","./en-sg":"NYST","./en-sg.js":"NYST","./eo":"Nd3h","./eo.js":"Nd3h","./es":"LT9G","./es-do":"7MHZ","./es-do.js":"7MHZ","./es-mx":"USNP","./es-mx.js":"USNP","./es-us":"INcR","./es-us.js":"INcR","./es.js":"LT9G","./et":"XlWM","./et.js":"XlWM","./eu":"sqLM","./eu.js":"sqLM","./fa":"2pmY","./fa.js":"2pmY","./fi":"nS2h","./fi.js":"nS2h","./fil":"rMbQ","./fil.js":"rMbQ","./fo":"OVPi","./fo.js":"OVPi","./fr":"tzHd","./fr-ca":"bXQP","./fr-ca.js":"bXQP","./fr-ch":"VK9h","./fr-ch.js":"VK9h","./fr.js":"tzHd","./fy":"g7KF","./fy.js":"g7KF","./ga":"U5Iz","./ga.js":"U5Iz","./gd":"nLOz","./gd.js":"nLOz","./gl":"FuaP","./gl.js":"FuaP","./gom-deva":"VGQH","./gom-deva.js":"VGQH","./gom-latn":"+27R","./gom-latn.js":"+27R","./gu":"rtsW","./gu.js":"rtsW","./he":"Nzt2","./he.js":"Nzt2","./hi":"ETHv","./hi.js":"ETHv","./hr":"V4qH","./hr.js":"V4qH","./hu":"xne+","./hu.js":"xne+","./hy-am":"GrS7","./hy-am.js":"GrS7","./id":"yRTJ","./id.js":"yRTJ","./is":"upln","./is.js":"upln","./it":"FKXc","./it-ch":"/E8D","./it-ch.js":"/E8D","./it.js":"FKXc","./ja":"ORgI","./ja.js":"ORgI","./jv":"JwiF","./jv.js":"JwiF","./ka":"RnJI","./ka.js":"RnJI","./kk":"j+vx","./kk.js":"j+vx","./km":"5j66","./km.js":"5j66","./kn":"gEQe","./kn.js":"gEQe","./ko":"eBB/","./ko.js":"eBB/","./ku":"kI9l","./ku.js":"kI9l","./ky":"6cf8","./ky.js":"6cf8","./lb":"z3hR","./lb.js":"z3hR","./lo":"nE8X","./lo.js":"nE8X","./lt":"/6P1","./lt.js":"/6P1","./lv":"jxEH","./lv.js":"jxEH","./me":"svD2","./me.js":"svD2","./mi":"gEU3","./mi.js":"gEU3","./mk":"Ab7C","./mk.js":"Ab7C","./ml":"oo1B","./ml.js":"oo1B","./mn":"CqHt","./mn.js":"CqHt","./mr":"5vPg","./mr.js":"5vPg","./ms":"ooba","./ms-my":"G++c","./ms-my.js":"G++c","./ms.js":"ooba","./mt":"oCzW","./mt.js":"oCzW","./my":"F+2e","./my.js":"F+2e","./nb":"FlzV","./nb.js":"FlzV","./ne":"/mhn","./ne.js":"/mhn","./nl":"3K28","./nl-be":"Bp2f","./nl-be.js":"Bp2f","./nl.js":"3K28","./nn":"C7av","./nn.js":"C7av","./oc-lnc":"KOFO","./oc-lnc.js":"KOFO","./pa-in":"pfs9","./pa-in.js":"pfs9","./pl":"7LV+","./pl.js":"7LV+","./pt":"ZoSI","./pt-br":"AoDM","./pt-br.js":"AoDM","./pt.js":"ZoSI","./ro":"wT5f","./ro.js":"wT5f","./ru":"ulq9","./ru.js":"ulq9","./sd":"fW1y","./sd.js":"fW1y","./se":"5Omq","./se.js":"5Omq","./si":"Lgqo","./si.js":"Lgqo","./sk":"OUMt","./sk.js":"OUMt","./sl":"2s1U","./sl.js":"2s1U","./sq":"V0td","./sq.js":"V0td","./sr":"f4W3","./sr-cyrl":"c1x4","./sr-cyrl.js":"c1x4","./sr.js":"f4W3","./ss":"7Q8x","./ss.js":"7Q8x","./sv":"Fpqq","./sv.js":"Fpqq","./sw":"DSXN","./sw.js":"DSXN","./ta":"+7/x","./ta.js":"+7/x","./te":"Nlnz","./te.js":"Nlnz","./tet":"gUgh","./tet.js":"gUgh","./tg":"5SNd","./tg.js":"5SNd","./th":"XzD+","./th.js":"XzD+","./tk":"+WRH","./tk.js":"+WRH","./tl-ph":"3LKG","./tl-ph.js":"3LKG","./tlh":"m7yE","./tlh.js":"m7yE","./tr":"k+5o","./tr.js":"k+5o","./tzl":"iNtv","./tzl.js":"iNtv","./tzm":"FRPF","./tzm-latn":"krPU","./tzm-latn.js":"krPU","./tzm.js":"FRPF","./ug-cn":"To0v","./ug-cn.js":"To0v","./uk":"ntHu","./uk.js":"ntHu","./ur":"uSe8","./ur.js":"uSe8","./uz":"XU1s","./uz-latn":"/bsm","./uz-latn.js":"/bsm","./uz.js":"XU1s","./vi":"0X8Q","./vi.js":"0X8Q","./x-pseudo":"e/KL","./x-pseudo.js":"e/KL","./yo":"YXlc","./yo.js":"YXlc","./zh-cn":"Vz2w","./zh-cn.js":"Vz2w","./zh-hk":"ZUyn","./zh-hk.js":"ZUyn","./zh-mo":"+WA1","./zh-mo.js":"+WA1","./zh-tw":"BbgG","./zh-tw.js":"BbgG"};function a(e){return n(i(e))}function i(e){var t=s[e];if(!(t+1))throw new Error("Cannot find module '"+e+"'.");return t}a.keys=function(){return Object.keys(s)},a.resolve=i,e.exports=a,a.id="uslO"}},["NHnr"]);
//# sourceMappingURL=app.d87dd85e1ad2eec9ea16.js.map