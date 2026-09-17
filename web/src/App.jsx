import React, { useMemo, useState } from "react";
import {
  Activity,
  Bot,
  Brain,
  ChevronDown,
  ChevronUp,
  ClipboardList,
  Database,
  Dumbbell,
  Footprints,
  Loader2,
  MessageSquare,
  Play,
  RotateCcw,
  Search,
  Settings2,
  Timer
} from "lucide-react";

const API_BASE_URL = "http://localhost:8000";

const SAMPLE_QUERIES = [
  "อยากทำ 10K sub-55 ตอนนี้วิ่งได้ 60 นาที มีเวลาซ้อม 3 วัน/สัปดาห์ ภายใน 8 สัปดาห์",
  "อยากวิ่ง 5K ให้ได้ 30 นาที ตอนนี้ 40 นาที ซ้อมได้ 4 วัน/สัปดาห์",
  "ช่วยแนะนำรองเท้าวิ่งสำหรับ easy run งบไม่เกิน 4000 บาท",
  "ช่วยแนะนำรองเท้าวิ่งสำหรับ easy run งบไม่เกิน 4000 บาท ของ nike หน่อย",
  "อยากวิ่ง 5K ให้ได้ 30 นาที ตอนนี้ 40 นาที ซ้อมได้ 4 วัน/สัปดาห์ และแนะนำรองเท้าสำหรับ tempo run ให้หน่อย"
];

function Badge({ children, tone = "default" }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function Card({ title, icon, children, right }) {
  return (
    <section className="card">
      <div className="card-header">
        <div className="card-title">
          {icon}
          <span>{title}</span>
        </div>
        {right}
      </div>
      <div className="card-body">{children}</div>
    </section>
  );
}

function EmptyState({ text }) {
  return <div className="empty-state">{text}</div>;
}

function JsonBlock({ data }) {
  if (!data || (Array.isArray(data) && data.length === 0)) {
    return <EmptyState text="ยังไม่มีข้อมูล debug ในส่วนนี้" />;
  }

  if (
    typeof data === "object" &&
    !Array.isArray(data) &&
    Object.keys(data).length === 0
  ) {
    return <EmptyState text="ยังไม่มีข้อมูล debug ในส่วนนี้" />;
  }

  return <pre className="json-block">{JSON.stringify(data, null, 2)}</pre>;
}

function RetrievedDocs({ title, docs, type }) {
  if (!docs || docs.length === 0) {
    return <EmptyState text={`ยังไม่มี ${title}`} />;
  }

  return (
    <div className="doc-list">
      {docs.map((doc, index) => (
        <div className="doc-item" key={`${type}-${index}`}>
          <div className="doc-topline">
            <strong>
              {type === "shoe" ? `Shoe Doc ${index + 1}` : `Coach Doc ${index + 1}`}
            </strong>

            {doc.distance_score !== undefined && (
              <Badge tone="muted">
                distance: {Number(doc.distance_score).toFixed(4)}
              </Badge>
            )}

            {doc.adjusted_score !== undefined && (
              <Badge tone="green">
                adjusted: {Number(doc.adjusted_score).toFixed(4)}
              </Badge>
            )}
          </div>

          <div className="doc-grid">
            {Object.entries(doc)
              .filter(([key]) => key !== "content" && key !== "preview")
              .map(([key, value]) => (
                <div className="doc-meta" key={key}>
                  <span>{key}</span>
                  <b>{String(value)}</b>
                </div>
              ))}
          </div>

          {(doc.preview || doc.content) && (
            <p className="doc-preview">{doc.preview || doc.content}</p>
          )}
        </div>
      ))}
    </div>
  );
}

function ChatHistory({ messages }) {
  if (!messages.length) {
    return <EmptyState text="ยังไม่มีประวัติการสนทนา" />;
  }

  return (
    <div className="history-list">
      {messages.map((msg, index) => (
        <div className={`history-item history-${msg.role}`} key={index}>
          <div className="history-role">
            {msg.role === "user" ? "User" : "Coach Agent"}
          </div>
          <div className="history-content">{msg.content}</div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [messages, setMessages] = useState([]);
  const [answer, setAnswer] = useState("");
  const [debug, setDebug] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showDebug, setShowDebug] = useState(true);
  const [debugOpen, setDebugOpen] = useState({
    plan: true,
    tools: true,
    docs: true,
    scores: true,
    constraints: true
  });
  const [error, setError] = useState("");

  const responseModeTone = useMemo(() => {
    const mode = debug?.response_mode;

    if (mode === "coach_only") return "blue";
    if (mode === "shoe_only") return "green";
    if (mode === "mixed") return "orange";

    return "muted";
  }, [debug]);

  const handleAsk = async () => {
    const trimmed = query.trim();

    if (!trimmed) {
      setError("กรุณาพิมพ์คำถามก่อน");
      return;
    }

    setLoading(true);
    setError("");

    const outgoingHistory = messages.slice(-8);

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: trimmed,
          chat_history: outgoingHistory
        })
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }

      const data = await res.json();

      const nextMessages = [
        ...messages,
        { role: "user", content: trimmed },
        { role: "assistant", content: data.answer || "" }
      ];

      setMessages(nextMessages);
      setAnswer(data.answer || "");
      setDebug(data);
    } catch (err) {
      setError(
        "เชื่อมต่อ backend ไม่สำเร็จ ตรวจสอบว่า FastAPI รันอยู่ที่ http://localhost:8000 แล้วลองใหม่"
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuery("");
    setMessages([]);
    setAnswer("");
    setDebug(null);
    setError("");
  };

  const toggleDebug = (key) => {
    setDebugOpen((prev) => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">
            <Activity size={16} />
            Agentic RAG Demo
          </div>

          <h1>Daniels' Running Formula</h1>

          <p>
            ระบบทดสอบ RAG สำหรับโค้ชวิ่ง แผนซ้อม 5K/10K และแนะนำรองเท้าวิ่ง
          </p>
        </div>

        <div className="status-card">
          <span className="status-dot" />
          <div>
            <small>AIE 455</small>
            <b>RAG Agent</b>
          </div>
        </div>
      </header>

      <main className="layout">
        <aside className="sidebar">
          <Card title="Sample Queries" icon={<ClipboardList size={18} />}>
            <div className="sample-list">
              {SAMPLE_QUERIES.map((sample, index) => (
                <button
                  key={index}
                  className="sample-button"
                  onClick={() => setQuery(sample)}
                >
                  <span>ตัวอย่าง {index + 1}</span>
                  <p>{sample}</p>
                </button>
              ))}
            </div>
          </Card>

          <Card title="Chat History" icon={<MessageSquare size={18} />}>
            <ChatHistory messages={messages} />
          </Card>
        </aside>

        <section className="main-panel">
          <Card
            title="Ask the Running Coach"
            icon={<Bot size={18} />}
            right={
              <button
                className="ghost-button"
                onClick={() => setShowDebug(!showDebug)}
              >
                <Settings2 size={16} />
                {showDebug ? "Hide Debug" : "Show Debug"}
              </button>
            }
          >
            <textarea
              className="query-box"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="พิมพ์คำถาม เช่น อยากทำ 10K sub-55..."
            />

            <div className="actions">
              <button
                className="primary-button"
                onClick={handleAsk}
                disabled={loading}
              >
                {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
                Ask Coach
              </button>

              <button
                className="secondary-button"
                onClick={handleClear}
                disabled={loading}
              >
                <RotateCcw size={18} />
                Clear
              </button>
            </div>

            {error && <div className="error-box">{error}</div>}
          </Card>

          <Card
            title="Coach Agent Answer"
            icon={<Dumbbell size={18} />}
            right={
              debug?.response_mode ? (
                <Badge tone={responseModeTone}>{debug.response_mode}</Badge>
              ) : null
            }
          >
            {answer ? (
              <div className="answer-box">{answer}</div>
            ) : (
              <EmptyState text="คำตอบจาก Agent จะแสดงตรงนี้" />
            )}
          </Card>

          {showDebug && (
            <div className="debug-panel">
              <div className="debug-header">
                <div>
                  <h2>Debug Details</h2>
                  <p>
                    แสดงรายละเอียดตาม requirement: action plan, tools, retrieved docs,
                    scores และ constraints
                  </p>
                </div>

                {debug?.note && <Badge tone="muted">{debug.note}</Badge>}
              </div>

              <div className="debug-grid">
                <Card title="Response Mode" icon={<Brain size={18} />}>
                  <div className="mode-row">
                    <Badge tone={responseModeTone}>
                      {debug?.response_mode || "N/A"}
                    </Badge>
                    <span>
                      coach_only = แผนซ้อม, shoe_only = รองเท้า, mixed = ทั้งสองอย่าง
                    </span>
                  </div>
                </Card>

                <Card title="Score Explanation" icon={<Timer size={18} />}>
                  <ul className="plain-list">
                    <li>
                      <b>distance_score:</b> ยิ่งต่ำ ยิ่งใกล้ query
                    </li>
                    <li>
                      <b>adjusted_score:</b> score หลัง rerank
                    </li>
                    <li>
                      <b>keyword_bonus:</b> คะแนนเพิ่มเมื่อเจอคำสำคัญที่ตรง
                    </li>
                    <li>
                      <b>keyword_penalty:</b> คะแนนลงโทษเมื่อเจอเนื้อหานอกประเด็น
                    </li>
                    <li>
                      <b>suitability_score:</b> ความเหมาะของรองเท้ากับประเภทการวิ่ง
                    </li>
                  </ul>
                </Card>
              </div>

              <Card
                title="Agent Action Plan"
                icon={<Brain size={18} />}
                right={
                  <button
                    className="collapse-button"
                    onClick={() => toggleDebug("plan")}
                  >
                    {debugOpen.plan ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                  </button>
                }
              >
                {debugOpen.plan && <JsonBlock data={debug?.action_plan} />}
              </Card>

              <Card
                title="Extracted Constraints"
                icon={<Search size={18} />}
                right={
                  <button
                    className="collapse-button"
                    onClick={() => toggleDebug("constraints")}
                  >
                    {debugOpen.constraints ? (
                      <ChevronUp size={18} />
                    ) : (
                      <ChevronDown size={18} />
                    )}
                  </button>
                }
              >
                {debugOpen.constraints && <JsonBlock data={debug?.constraints} />}
              </Card>

              <Card
                title="Tool Calls"
                icon={<Settings2 size={18} />}
                right={
                  <button
                    className="collapse-button"
                    onClick={() => toggleDebug("tools")}
                  >
                    {debugOpen.tools ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                  </button>
                }
              >
                {debugOpen.tools && <JsonBlock data={debug?.tool_calls} />}
              </Card>

              <Card
                title="Retrieved Documents"
                icon={<Database size={18} />}
                right={
                  <button
                    className="collapse-button"
                    onClick={() => toggleDebug("docs")}
                  >
                    {debugOpen.docs ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                  </button>
                }
              >
                {debugOpen.docs && (
                  <div className="retrieval-tabs">
                    <div>
                      <h3>Coach Retrieved Docs</h3>
                      <RetrievedDocs
                        title="Coach Retrieved Docs"
                        docs={debug?.retrieved_coach_docs}
                        type="coach"
                      />
                    </div>

                    <div>
                      <h3>Shoe Retrieved Docs</h3>
                      <RetrievedDocs
                        title="Shoe Retrieved Docs"
                        docs={debug?.retrieved_shoe_docs}
                        type="shoe"
                      />
                    </div>
                  </div>
                )}
              </Card>

              <Card title="Shoe / Coach Debug Raw" icon={<Footprints size={18} />}>
                <JsonBlock data={debug} />
              </Card>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}