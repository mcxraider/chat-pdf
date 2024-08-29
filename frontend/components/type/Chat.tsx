export default function Chat() {
  return (
    <div
      style={{
        flex: 1,
        padding: "1rem",
        backgroundColor: "#eaeaea",
        overflowY: "auto",
      }}
    >
      <div style={{ marginBottom: "1rem" }}>
        <div
          style={{
            backgroundColor: "#0070f3",
            color: "white",
            padding: "10px",
            borderRadius: "10px",
            width: "fit-content",
            marginBottom: "10px",
          }}
        >
          Hello! How are you?
        </div>
        <div
          style={{
            backgroundColor: "#f1f0f0",
            padding: "10px",
            borderRadius: "10px",
            width: "fit-content",
            marginBottom: "10px",
            alignSelf: "flex-end",
          }}
        >
          good, thanks! How about you?
        </div>
      </div>
    </div>
  );
}
